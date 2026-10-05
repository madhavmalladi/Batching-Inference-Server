# batchserve: a dynamic-batching inference server

A small version of NVIDIA Triton for object detection. Clients send one image per
gRPC call. The gateway groups concurrent requests into batches, runs YOLOv8n once
per batch on a C++ ONNX Runtime worker, and sends each result back to the right
client. The goal of the project is to **measure the throughput vs. latency
tradeoff** of dynamic batching.

This is a learning and portfolio project. Prefer simple, well-understood code over
clever code. Every non-obvious decision gets a short comment explaining *why*.

Deeper reasoning lives in `docs/`: `design.md` (why it's built this way),
`fake-worker.md`, `test-plan.md`, and `agents.md` (task split and integration risks).

## Architecture

```
client --Detect(image)--> gateway (Go) --InferBatch([images])--> worker (C++, ONNX Runtime)
                          |  bounded queue                         decode, letterbox, Run, NMS
                          |  single batcher goroutine
                          |  /metrics --> Prometheus
```

- **gateway/** (Go): public `Detector` service, admission, bounded queue, batcher,
  dispatch to the worker, Prometheus metrics. Also contains `cmd/fakeworker`.
- **worker/** (C++17): `InferenceWorker` service. Takes a batch of encoded images
  and returns detections per image, in order. CPU execution provider by default;
  CUDA is optional and untested (no GPU available).
- **loadgen/** (Go + Python): Go load generator (open-loop and closed-loop),
  Python sweep and plotting scripts.
- **deploy/**: Dockerfiles, docker-compose, Prometheus config.

## The contract (FROZEN)

`proto/batchserve/v1/{detector,worker}.proto` and the generated Go code in `gen/go/`
are the contract every component builds against. **No component agent may edit
`proto/`, `gen/`, `docs/`, this file, the root `Makefile`, or `.gitignore`.** If you
need a contract change, write it up in `<your-dir>/CONTRACT_CHANGE.md` (what, why,
who is affected), stop, and tell the user.

Key contract rules (details are in the proto comments):
- Client deadlines travel as gRPC deadlines (from `ctx`), not as message fields.
- `InferBatchResponse.results` has the same length and order as the request, and
  echoes `request_id`.
- A bad image returns a per-image `error`; it does not fail the batch. A non-OK
  RPC status means the whole batch failed.
- Boxes are in original-image pixel coordinates.

## Batching rules (gateway)

Parameters: `max_batch_size` (B), `max_wait` (W), `queue_size` (Q), `max_inflight_batches` (F).

1. **Admission** (in the request's own handler goroutine):
   - Empty image, or larger than `--max-image-bytes`: `INVALID_ARGUMENT`.
   - ctx already done (deadline passed or cancelled): `DEADLINE_EXCEEDED` / `CANCELLED`.
   - Non-blocking send into the queue (a buffered `chan` of capacity Q). If it
     is full, return `RESOURCE_EXHAUSTED` **immediately**. Never block on admission.
2. **Batch formation**: one batcher goroutine.
   - First acquire an in-flight slot (a semaphore of size F). Then block until
     the first request arrives.
   - The batch deadline is `first.enqueuedAt + W`. It counts from when the
     **oldest request was admitted**, not from when the batcher saw it. Under a
     backlog it may already be in the past.
   - Without blocking, take requests already queued until the batch has B. Then
     wait for more requests or the batch deadline, whichever comes first.
     Dispatch on size == B (`FULL`) or on the deadline (`TIMEOUT`).
   - W = 0 means: dispatch whatever is immediately available. B = 1 means no
     batching (this is the baseline).
   - Skip requests whose ctx is already done (lazy deletion, counted as expired).
     If every request was skipped, send nothing and release the slot.
3. **Dispatch**: one goroutine per batch calls `InferBatch` with a fixed
   `--worker-timeout`. The call does **not** use the minimum client deadline:
   one impatient client must not kill the whole batch. Results are matched to
   requests by position (and the `request_id` echo is checked). Each one is
   written to that request's own `chan result` of **capacity 1**, so the write
   never blocks even if the client has already left. Then the slot is released.
4. **Waiting handler**: `select` on its result channel and `ctx.Done()`. A client
   whose deadline passes gets `DEADLINE_EXCEEDED` at its deadline, not when the
   batch finishes.
5. **Backpressure path**: worker slow → slots stay busy → batcher stops pulling →
   queue fills → new requests get `RESOURCE_EXHAUSTED`. The most work the system
   holds at once is Q + B×F requests.

Error mapping: queue full → `RESOURCE_EXHAUSTED`; per-image worker error →
`INVALID_ARGUMENT`; worker RPC error or timeout → `UNAVAILABLE`; wrong result
count or ID mismatch → `INTERNAL`; shutting down → `UNAVAILABLE`.

## Metrics (gateway, `/metrics`; names are a contract)

| Name | Type | Labels |
|---|---|---|
| `batchserve_queue_depth` | gauge (len of queue chan) | |
| `batchserve_batch_size` | histogram, buckets 1,2,4,8,16,32,64 | |
| `batchserve_batches_total` | counter | `reason` = full\|timeout |
| `batchserve_request_duration_seconds` | histogram (admission → response) | `code` (gRPC code name) |
| `batchserve_queue_wait_seconds` | histogram | |
| `batchserve_batch_duration_seconds` | histogram (InferBatch round trip) | |
| `batchserve_worker_inference_seconds` | histogram (worker-reported `inference_us`) | |
| `batchserve_requests_rejected_total` | counter | `reason` = queue_full\|invalid |
| `batchserve_requests_expired_total` | counter | `stage` = admission\|queued\|inflight |

p50 and p99 are computed in Prometheus with `histogram_quantile`. Latency buckets
are exponential from 0.5 ms to about 10 s.

## Ports and flags (a contract: deploy/ and loadgen/ depend on these)

| Process | Port | Flags (defaults) |
|---|---|---|
| gateway | gRPC `:50051`, metrics `:2112` | `--listen=:50051 --metrics-listen=:2112 --worker-addr=localhost:50052 --max-batch-size=8 --max-wait=5ms --queue-size=256 --max-inflight-batches=1 --worker-timeout=10s --max-image-bytes=2097152` |
| worker | `:50052` | `--listen=0.0.0.0:50052 --model=models/yolov8n.onnx --provider=cpu --intra-op-threads=0 --conf-threshold=0.25 --iou-threshold=0.45 --max-detections=100` |
| fakeworker | `:50052` | `--listen=:50052 --base-latency=5ms --per-image-latency=2ms --jitter=0 --fail-rate=0 --seed=1` |
| prometheus | `:9090` | scrapes `gateway:2112` every 1s |
| loadgen | — | `--target=localhost:50051 --mode=open\|closed --rate --concurrency --duration --warmup --deadline --images --out` |

`--max-wait` and other durations use Go `time.Duration` syntax. The gateway
rejects `--max-batch-size` values outside 1..64. The worker accepts gRPC
messages up to 256 MiB (the default 4 MiB is too small for a batch of images).

## Repo layout

```
proto/batchserve/v1/   FROZEN .proto contract
gen/go/                FROZEN generated Go stubs (module batchserve/gen/go), via `make proto`
docs/                  design, fake-worker spec, test plan, agent tasks
gateway/   [Agent A]   Go module batchserve/gateway
  cmd/gateway, cmd/fakeworker, internal/{batcher,server,workerclient,metrics,fakeworker}
worker/    [Agent B]   C++17 + CMake. src/, tests/, third_party/stb_image.h,
                       scripts/export_model.sh, models/ (gitignored *.onnx)
loadgen/   [Agent C]   Go module batchserve/loadgen (cmd/loadgen, internal/...)
                       bench/ (Python: sweep.py, plot.py, tests/), charts/ (committed PNGs)
deploy/    [Agent D]   gateway.Dockerfile (targets: gateway, fakeworker), worker.Dockerfile,
                       docker-compose.yml, prometheus/prometheus.yml, smoke_test.sh
```

Go modules import the stubs as `batchserve/gen/go/batchservev1` and need a
`replace batchserve/gen/go => ../gen/go` line in their `go.mod`. The C++ worker
generates its own C++ stubs from `proto/` at CMake configure/build time and never
commits them.

## Build and test

Every component directory has a `Makefile` with `build`, `test`, and `clean` targets.

```
make proto                # regenerate gen/go (only during contract changes)
make test                 # runs `make -C <dir> test` for every component that exists

cd gateway && make test   # go vet ./... && go test -race ./...
cd worker  && make test   # cmake -B build && cmake --build build && ctest --test-dir build
cd loadgen && make test   # go test -race ./... && python -m pytest bench/tests
cd deploy  && make test   # docker compose config && ./smoke_test.sh (uses fakeworker)
```

macOS native worker deps: `brew install onnxruntime grpc protobuf`. **Watch out:**
`/opt/anaconda3/bin/protoc` (3.20) comes earlier on PATH than Homebrew's protoc.
CMake must use the protoc that matches the libprotobuf it links
(`find_package(Protobuf CONFIG)` / `$(brew --prefix)/bin/protoc`), not whatever
is on PATH.

## Coding conventions

- **Go**: gofmt and `go vet` clean. Standard library first: `flag`, `log/slog`,
  `testing`. Use `testing/synctest` for any timing-dependent test, so tests are
  deterministic and run in virtual time. Never use `time.Sleep` in tests.
  Tests run with `-race`. Errors are wrapped with `%w`. No global mutable state
  except Prometheus registries, which are injected so tests can use a fresh
  registry. Short packages, and an interface only where there is a real second
  implementation (e.g. `Inferer`: gRPC client vs. in-memory fake).
- **C++**: C++17, `-Wall -Wextra` on our own targets, RAII, no raw `new`/`delete`.
  Use `std::vector<float>` for tensors. GoogleTest via CMake `FetchContent`.
  Readability beats speed: the worker must stay small enough to read in one
  sitting. Comment the shape of every tensor (`// [N, 3, 640, 640]`).
- **Python**: 3.12, plain `venv` + `requirements.txt`, pytest, matplotlib.
  Scripts take argparse flags and never hardcode paths.
- Tests are named as in `docs/test-plan.md`. Add more tests freely; don't rename those.
- Commit messages: `<component>: <imperative summary>`, e.g. `gateway: add batcher timeout path`.

## Multi-agent rules

1. **Each agent edits only its own directory** (gateway/, worker/, loadgen/, or deploy/).
   Read anything; write only there.
2. The frozen files listed above change only through `CONTRACT_CHANGE.md` plus a user decision.
3. Don't depend on another component's internals. Depend only on the proto, the
   port/flag table, and the metric names above.
4. Before you call your work done: `make test` in your directory passes, and the
   acceptance criteria in `docs/agents.md` for your agent are met.
