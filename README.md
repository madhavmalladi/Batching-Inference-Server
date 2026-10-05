# batchserve

A dynamic-batching inference server for object detection, in the spirit of NVIDIA
Triton Inference Server but small enough to read in a weekend.

Clients send **one image per gRPC call**. The server groups concurrent requests
into batches, runs YOLOv8n **once per batch**, and routes each result back to the
client that sent it. The point of the project is to **measure the throughput vs.
latency tradeoff** that batching creates.

> **Status:** planning complete, implementation not started. The contract
> (`proto/`), design docs (`docs/`), and test plan are written. Benchmark charts
> will appear in the [Results](#results) section once the components are built.

## Why batching?

Running a model has a large fixed cost per call (scheduling, weight reads, thread
or kernel-launch overhead). A single image leaves most of the hardware idle. A
batch of N pays that fixed cost once:

    per-image cost ≈ fixed / N + per_image

The price is latency: a request may wait for others to arrive, and bigger
batches take longer to run.

## Architecture

```
client ──Detect(image)──▶ gateway (Go) ──InferBatch([images])──▶ worker (C++)
                          │ bounded queue                        decode, letterbox,
                          │ size-or-timeout batcher              ONNX Runtime, NMS
                          └─ /metrics ──▶ Prometheus
```

| Component | Dir | Language | Role |
|---|---|---|---|
| Gateway / scheduler | `gateway/` | Go | gRPC server, bounded queue, dynamic batcher, backpressure, deadlines, Prometheus metrics. Also contains a fake worker for testing. |
| Inference worker | `worker/` | C++17 | ONNX Runtime + YOLOv8n: preprocess, one `Run()` per batch, NMS. CPU by default; CUDA optional (untested). |
| Load generator | `loadgen/` | Go + Python | Open-loop and closed-loop traffic, parameter sweeps, charts. |
| Infrastructure | `deploy/` | Docker, compose | Dockerfiles, docker-compose, Prometheus config. |

### Batching policy

A batch is dispatched when it reaches `max_batch_size` **or** when the oldest
queued request has waited `max_wait`, whichever comes first. Both are
configurable.

### Backpressure and deadlines

- The queue is bounded. When it is full, new requests are rejected immediately
  with `RESOURCE_EXHAUSTED` instead of queueing without bound.
- Requests whose client deadline has expired are dropped before they reach the
  worker.
- Each request carries its own reply channel, so every result goes back to the
  client that sent it.

The reasoning behind these choices is in [`docs/design.md`](docs/design.md).

## Repo layout

```
proto/     gRPC contract (frozen): client<->gateway and gateway<->worker
gen/go/    generated Go stubs, produced by `make proto`
gateway/   Go gateway, batcher, fake worker
worker/    C++ ONNX Runtime inference worker
loadgen/   load generator, benchmark sweeps, charts
deploy/    Dockerfiles, docker-compose, Prometheus config
docs/      design notes, fake-worker spec, test plan, agent task breakdown
```

## Getting started

> Components are not built yet; these commands will work once they are.

```bash
# Run the whole stack with the fake worker (no model needed)
cd deploy && make up-fake

# Or run natively
./gateway/bin/fakeworker --base-latency=5ms --per-image-latency=2ms &
./gateway/bin/gateway --max-batch-size=8 --max-wait=5ms &
./loadgen/bin/loadgen --mode=open --rate=200 --duration=20s --images=loadgen/images
```

Prerequisites: Go 1.26+, `protoc`, CMake, and (for the real worker)
`brew install onnxruntime grpc protobuf` plus a YOLOv8n ONNX export
(`worker/scripts/export_model.sh`).

## Testing

```bash
make test             # every component that exists
cd gateway && make test
```

Timing-dependent gateway tests run in virtual time with `testing/synctest`, so
they are deterministic. The full list of tests is in
[`docs/test-plan.md`](docs/test-plan.md).

## Results

_Charts will be added after the benchmark sweeps run on the real worker:_

- Throughput vs. p99 latency, one line per `max_batch_size` (batch size 1 as the baseline)
- Latency vs. offered load (shows where the system saturates)
- Batch size distribution
- Effect of `max_wait` on latency and throughput

Benchmarks run on CPU only (Apple Silicon), natively rather than in Docker.
Expect batching gains of roughly 1.5-3x on CPU, smaller than on a GPU, though
the shape of the tradeoff is the same.

## Docs

- [`CLAUDE.md`](CLAUDE.md): project rules, batching rules, ports, flags, conventions
- [`docs/design.md`](docs/design.md): why it is built this way
- [`docs/fake-worker.md`](docs/fake-worker.md): fake worker spec
- [`docs/test-plan.md`](docs/test-plan.md): tests per component
- [`docs/agents.md`](docs/agents.md): parallel agent tasks and integration risks
