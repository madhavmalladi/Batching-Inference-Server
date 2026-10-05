# Gateway walkthrough

How to read `gateway/`, in order, and what to be careful about. The git history
is also readable in order (`git log --reverse`): fake worker → metrics → batcher
policy → deadlines → server tests → shutdown → metrics tests → binaries → integration.

## Reading order

1. **`internal/batcher/batcher.go`**: the whole point of the project. Read it top to bottom.
   - The package comment draws the goroutine picture: handlers → queue → one batcher → one dispatch goroutine per batch.
   - `Submit` is what each request's own goroutine runs: admission, then wait.
   - `Run` + `formBatch` are the batching policy (~90 lines with comments). Each of the five rules in CLAUDE.md is marked `RULE n` or "Deadline check n of 3".
   - `dispatch` sends one batch and writes the results. `route` validates the worker's answer and maps errors to gRPC codes.
2. **`internal/batcher/batcher_test.go`**: the policy as executable examples. Read two or three tests (`BurstLargerThanBatchSplits`, `BacklogDispatchesWithoutWaiting`, `AcquiresSlotBeforeFormingBatch`). All run in virtual time, so "t=10ms" is exact.
3. **`internal/batcher/deadline_test.go`, `shutdown_test.go`**: expiry and shutdown behaviour.
4. **`internal/fakeworker/fakeworker.go`**: the stand-in worker. It also shows the test hooks (`Block`, `Calls`, `MaxConcurrent`) the batcher tests rely on.
5. **`internal/server/server.go`**: the public `Detector` handler: validation, then `Submit`, plus one place that records duration and status code. `Register` also adds health and reflection.
6. **`internal/server/{admission,deadline,routing}_test.go`**: backpressure, deadlines, result routing, `ServerTiming`.
7. **`internal/metrics/metrics.go`** and its tests. Tests drive the real batcher and read the registry.
8. **`internal/workerclient/workerclient.go`**: the gRPC `Inferer` (256 MiB messages, fast reconnect).
9. **`cmd/gateway/main.go`**: flags, wiring and the shutdown order. `cmd/fakeworker/main.go` is a thin wrapper.
10. **`internal/testrig/testrig.go`**: test-only wiring shared by the test packages. **`internal/integration`**: real sockets, real time.

## What each file does

| File | Role |
|---|---|
| `internal/batcher/batcher.go` | Queue, batch formation, in-flight limit, dispatch, result routing, shutdown |
| `internal/fakeworker/fakeworker.go` | `InferenceWorker` that sleeps and returns content-derived detections |
| `internal/server/server.go` | `Detector.Detect`: validation, status codes, request metrics; health + reflection |
| `internal/metrics/metrics.go` | Prometheus collectors (names and labels are a contract) and the `/metrics` handler |
| `internal/workerclient/workerclient.go` | `batcher.Inferer` over gRPC |
| `internal/testrig/testrig.go` | Wires fake worker + batcher + handler + metrics for tests |
| `internal/integration/integration_test.go` | End-to-end, startup order, throughput comparison, graceful shutdown |
| `cmd/gateway`, `cmd/fakeworker` | Binaries (flags match the CLAUDE.md table) |

## The three decisions a reader is most likely to get wrong

### 1. The batcher takes an in-flight slot *before* it takes a request, and the batch deadline starts at the oldest request's *admission*

Both rules look backwards at first, and both are covered by tests that fail if you "fix" them.

- **Slot first** (`Run`). The natural code is "form a batch, then wait for a worker slot". But a batch closed before the slot is free can't absorb requests that arrive while waiting, so under load you send small batches while a queue builds behind them. Taking the slot first leaves requests in the queue, and the next batch is full. Test: `AcquiresSlotBeforeFormingBatch`.
- **Deadline from admission** (`formBatch`, `first.enqueuedAt.Add(MaxWait)`). Using `time.Now()+W` seems equivalent, but under a backlog it makes every queued request wait an extra W. With the admission time the deadline is already past and the batch goes out at once. Test: `BacklogDispatchesWithoutWaiting`.
- **Related trap:** `formBatch` drains the queue *without blocking* before it starts the timer. If both a ready request and an expired timer were in one `select`, Go would choose randomly and a backlogged batch would sometimes leave work behind.

### 2. `Detect` returns at the client's deadline, but the batch is never cancelled and the late result is never lost

- `Submit` selects on `p.result` and `ctx.Done()`. An impatient client leaves at its own deadline while its batch keeps running, because its batch-mates still want their answers. That's why the worker call uses a **fixed** `--worker-timeout` and a `Background`-derived context, not the minimum client deadline or the request's context.
- **`result` has capacity 1.** The dispatcher then never blocks writing to a client that has gone. With an unbuffered channel, one abandoned request would leak a goroutine and hold the batch's slot forever. `TestDetect_LateResultDoesNotLeakGoroutine` proves it: if the send could block, the test's shutdown would deadlock.
- **Expired requests are deleted lazily.** A channel can't remove from the middle, so an expired request keeps its queue slot until it reaches the front, where `live()` skips it. The cost is that Q can fill with dead requests. The three `expired_total` stages are counted in three different places on purpose: `admission` in `Submit`, `queued` in `live()` (the batcher), and `inflight` in `dispatch` after the result is written. A request is counted at most once.

### 3. Shutdown order: stop the gRPC server first, *then* the batcher

`cmd/gateway/main.go` gives the batcher its own context instead of the signal context. On SIGTERM it runs `GracefulStop()` (no new RPCs; waits for running handlers) *while the batcher is still alive*, so handlers waiting in the queue get real answers. Only then does it cancel the batcher. Cancelling the batcher first would fail every waiting request with `UNAVAILABLE`.

Inside `Run`, the same idea applies at a smaller scale. Batches already at the worker finish. Anything not yet sent is failed immediately (`failUnsent`). A request that sneaks into the queue after that final drain is woken by `stopped`, which is closed only after the in-flight batches are done (defer order matters: `close(stopped)` is deferred first, so it runs last).

## Smaller things that will surprise you

- **`request_duration_seconds{code}` uses `RESOURCE_EXHAUSTED`-style names**, not grpc-go's `ResourceExhausted`. CLAUDE.md says "gRPC code name" without spelling it out; I followed the proto comments. If loadgen or deploy queries assume the CamelCase form, this is the one place to change (`metrics.codeNames`).
- **The queue-depth gauge is a `GaugeFunc`** reading `len(queue)` at scrape time, so it can't drift. That's why `batcher.New` calls `metrics.BindQueueDepth`.
- **Labelled metrics are pre-created** so every series appears on `/metrics` at 0 before any traffic. Without that, a `CounterVec` exports nothing until first use.
- **`testrig` is not test-only by file name**: it is a normal package (it must be, to be shared across test packages) but nothing in `cmd/` imports it.
- **Tests never call `time.Sleep`.** Where virtual time must advance, they wait on `time.After` inside a `synctest` bubble. The integration tests use real time and poll with a ticker; `-short` skips the one slow test.
- **The most the system can hold at once is Q + B×F requests.** `TestDetect_QueueOverflow…` is built on exactly that arithmetic (1 in flight + 4 queued, 6th rejected).
- **Fake-worker numbers are not performance numbers** (see `docs/fake-worker.md`). `TestIntegration_BatchingBeatsNoBatching` only checks the *shape*: B=8 beats B=1 by more than 1.5×.
