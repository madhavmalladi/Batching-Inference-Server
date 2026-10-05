# Fake worker spec

A Go implementation of `InferenceWorker` that sleeps instead of running a model.
It lets the gateway, load generator, and compose stack be built and tested
before the C++ worker exists. Owned by **Agent A**.

- Logic lives in `gateway/internal/fakeworker`: a plain Go type that the gateway
  tests use in-process, with no network.
- The binary is `gateway/cmd/fakeworker`, a thin gRPC wrapper around the same type.
- Docker: the `fakeworker` target in `deploy/gateway.Dockerfile` (built by Agent D).

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--listen` | `:50052` | Same port as the real worker, so the gateway's defaults work with either. |
| `--base-latency` | `5ms` | Fixed cost per batch. |
| `--per-image-latency` | `2ms` | Added cost per image. |
| `--jitter` | `0` | Fraction, e.g. `0.1` = uniformly random ±10% of the total sleep. |
| `--fail-rate` | `0` | Probability that a whole batch returns `UNAVAILABLE`. |
| `--seed` | `1` | RNG seed for jitter and fail-rate (reproducible runs). |

## Behavior

**Latency model:** `sleep = (base + per_image × N) × (1 + U(−jitter, +jitter))`.

With the defaults, per-image cost is 7 ms at N=1 and about 2.6 ms at N=8. That
makes batching visibly worthwhile, just like the real fixed-plus-per-image cost.
The sleep respects ctx: if the RPC's ctx ends during the sleep, it returns
`ctx.Err()` as the gRPC status right away.

**Deterministic, content-derived detections.** Each result depends only on that
image's bytes, so routing tests can check that every client got *its own* answer:

| Field | Value |
|---|---|
| `request_id` | echoed |
| detections | exactly 1 |
| `class_id` | `crc32.ChecksumIEEE(data) % 80` |
| `class_name` | `"class_<class_id>"` |
| `score` | `0.5` |
| `box` | `{x_min: 0, y_min: 0, x_max: float(len(data)), y_max: float(len(data))}` |

**Bad images:** if `data` is empty or starts with the ASCII bytes `BADIMG`, that
image's result has `error = "fake decode error"` and no detections. The rest of
the batch is unaffected.

**Ordering:** results are in request order with the same length. No real image
decoding happens, so any bytes are accepted (useful for synthetic loadgen payloads).

**Health:** registers `grpc.health.v1` and reports `SERVING` immediately.

**Observability for tests:** the in-process type records each batch it receives:
`Calls() []BatchRecord{Size int; IDs []uint64; At time.Time}`. It also has
`MaxConcurrent() int` for max-inflight tests, and an optional `Block chan struct{}`
hook. When `Block` is set, `InferBatch` waits on it before sleeping. Tests use
this to hold the worker "busy" and fill the queue.

**Logs:** one `slog` line per batch: `batch_id`, `size`, `slept_ms`.

## Things the fake does NOT model (so tuning on it can mislead)

- CPU contention between concurrent batches. The fake scales perfectly with
  max_inflight; the real CPU worker does not.
- Decode cost that grows with image size.
- A per-image cost that is non-linear in batch size (cache effects; the real
  curve flattens).

Use the fake to check **correctness and policy behavior**. Get **performance
numbers** from the real worker.
