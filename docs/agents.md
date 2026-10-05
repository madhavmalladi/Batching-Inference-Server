# Parallel agent tasks

## Step 0: before any agent starts (on `main`)

1. `git init`. Commit `proto/`, `gen/go/go.mod`, `docs/`, `CLAUDE.md`, `Makefile`, and `.gitignore`.
2. `make proto`. This generates `gen/go/batchservev1/*.pb.go` and tidies `gen/go/go.mod`. Commit it.
   Both Go agents now import identical stubs, and neither one owns them.
3. Create one worktree per agent:
   ```
   git worktree add ../bis-gateway -b agent-a/gateway
   git worktree add ../bis-worker  -b agent-b/worker
   git worktree add ../bis-loadgen -b agent-c/loadgen
   git worktree add ../bis-deploy  -b agent-d/deploy
   ```
4. Start one Claude Code session in each worktree with the kickoff prompt below.

Dependency graph: every agent depends only on step 0, so **all four can start at once.**
Each agent replaces the components it doesn't own with a stub (see "Stubs" in each section).

```
step 0 ──┬── A gateway+fakeworker ──┐
         ├── B worker ──────────────┤
         ├── C loadgen ─────────────┼──▶ merge: A → C → B → D ──▶ benchmarks + README
         └── D deploy ──────────────┘
```

---

## Agent A: gateway + batcher + fake worker

- **Owns:** `gateway/` (Go module `batchserve/gateway`).
- **Inputs:** `proto/`, `gen/go`, the batching rules, metrics, and flags in CLAUDE.md, `docs/fake-worker.md`, `docs/design.md` §3–7.
- **Outputs:** `gateway` and `fakeworker` binaries. `make build` puts them in `gateway/bin/`.
- **Stubs:** none needed. The fake worker *is* its stub for the worker.
- **Suggested internal shape:**
  - `internal/batcher`: `Batcher{queue chan *pending, sem chan struct{}, inferer Inferer, ...}`, `Submit(ctx, img) (result, error)`, and `Run(ctx)`
  - `internal/workerclient`: an `Inferer` implemented over gRPC (max send size 256 MiB, fixed timeout)
  - `internal/server`: the `Detector` gRPC handler (validation and status mapping)
  - `internal/metrics`: a struct of collectors built against an injected `prometheus.Registerer`
  - `internal/fakeworker`: the fake's logic. `cmd/fakeworker` is a thin gRPC wrapper around it.
- **Acceptance criteria:**
  - [ ] Every section-A test in `docs/test-plan.md` exists and passes with `go test -race ./...`.
  - [ ] Batcher tests use `testing/synctest`, with no `time.Sleep` in tests.
  - [ ] Every flag, default, port, and metric name exactly matches CLAUDE.md.
  - [ ] Manual check: running `fakeworker` and `gateway`, then calling `grpcurl -plaintext -d '{"image":"aGVsbG8="}' localhost:50051 batchserve.v1.Detector/Detect` returns a detection with `x_max: 5`. (This needs gRPC reflection enabled on the gateway, which is cheap and makes debugging easy.)
  - [ ] `curl localhost:2112/metrics` shows every `batchserve_*` metric.
  - [ ] The batcher's core loop is under ~150 lines and has a comment explaining each rule.

**Kickoff prompt:**
> You are Agent A. Read CLAUDE.md, docs/design.md, docs/fake-worker.md, and section A of docs/test-plan.md. Implement the gateway and fake worker in `gateway/` only. Start with `internal/fakeworker` and the batcher tests (TDD), then the batcher, server, metrics, and the cmd binaries. Do not edit anything outside `gateway/`. If the contract seems wrong, write `gateway/CONTRACT_CHANGE.md` and stop. Explain your concurrency decisions in code comments: I'm learning from this code.

---

## Agent B: C++ inference worker

- **Owns:** `worker/`.
- **Inputs:** `proto/` (generates its own C++ stubs at build time), CLAUDE.md flags and ports, and `docs/design.md` §9.
- **Outputs:** the `worker/build/worker` binary, `scripts/export_model.sh` (creates a venv, runs `pip install ultralytics`, and runs `yolo export ... dynamic=True opset=17` into `models/`), and `scripts/fetch_testdata.sh` (downloads `bus.jpg`).
- **Stubs:** none. It is tested on its own with GoogleTest and grpcurl.
- **Suggested files:** `src/main.cpp` (flags, wiring), `src/server.{h,cpp}` (gRPC service and health), `src/image.{h,cpp}` (decode and letterbox), `src/detector.{h,cpp}` (ORT session wrapper and batch tensor), `src/postprocess.{h,cpp}` (decode output and NMS), `src/coco_classes.h`, and `third_party/stb_image.h`.
- **Build:** a single `CMakeLists.txt` that works with Homebrew on macOS and in the Debian container. ORT is found via `ONNXRUNTIME_ROOT` (default `$(brew --prefix onnxruntime)`). CUDA sits behind `--provider=cuda` and a CMake option `WITH_CUDA=OFF`. That path is written but marked untested in the code.
- **Acceptance criteria:**
  - [ ] Every section-B test exists. The non-★ tests pass with no model present, and the ★ tests pass after running `export_model.sh` and `fetch_testdata.sh`.
  - [ ] `DetectorTest.BatchMatchesSingleRuns` passes.
  - [ ] The worker refuses to start on a model with a fixed batch dimension, and logs the input shape.
  - [ ] Health reports SERVING only after the model has loaded.
  - [ ] grpcurl InferBatch with two images returns two ordered results.
  - [ ] Excluding tests and third_party, `src/` is roughly 600 lines or fewer. Every tensor has a shape comment.
  - [ ] Builds without warnings from our own code under `-Wall -Wextra`. Tests also pass under `-DSANITIZE=ON` (ASan and UBSan).

**Kickoff prompt:**
> You are Agent B. Read CLAUDE.md, docs/design.md §8–9, and section B of docs/test-plan.md. Implement the C++ inference worker in `worker/` only. Build the pure functions (decode, letterbox, postprocess, NMS) test-first, then the ORT detector, then the gRPC server. Keep it small and heavily commented: I need to understand every line. Use the Homebrew protoc that matches libprotobuf, not the anaconda one on PATH. Do not edit outside `worker/`.

---

## Agent C: load generator and benchmarks

- **Owns:** `loadgen/` (Go module `batchserve/loadgen`, plus `bench/` in Python).
- **Inputs:** `gen/go`, the CLAUDE.md flag table (to launch the gateway in sweeps), the metric names, and `ServerTiming` in the response.
- **Outputs:**
  - The `loadgen` binary, which writes `requests.csv` (send_ts, latency_us, code, batch_size, queue_wait_us, dispatch_reason) and `summary.json` (offered rate, achieved throughput, p50/p90/p99/max, counts by code).
  - `bench/sweep.py` takes a YAML/JSON config. For each (B, W) it starts `gateway` as a subprocess with those flags against an already-running worker, waits for readiness, and runs loadgen at each rate.
  - `bench/plot.py` produces `charts/throughput_vs_p99.png` (one line per B, with B=1 as the baseline), `charts/latency_vs_offered_load.png` (shows the knee), `charts/batch_size_distribution.png`, and `charts/max_wait_tradeoff.png`.
  - `scripts/fetch_images.sh` downloads about 50 COCO val2017 images to `loadgen/images/` (gitignored).
- **Stubs:** a tiny in-process `Detector` server in `internal/stubserver` (for tests only). For real runs before merge, it can use Agent A's fakeworker by building from the A branch, but nothing from A is committed here.
- **Suggested sweep defaults:** Sweep 1 fixes W=5ms and varies B ∈ {1,2,4,8,16,32} × 6 rates. Sweep 2 fixes B=16 and varies W ∈ {0,1,2,5,10,20}ms × 6 rates. Each point is 5s warmup plus 20s measured.
- **Acceptance criteria:**
  - [ ] Every section-C test exists and passes.
  - [ ] Open-loop mode is the default. The README section that Agent C drafts (in `loadgen/README.md`) explains coordinated omission.
  - [ ] A sweep against the fakeworker produces all four charts from scratch with one command: `make bench-fake`.
  - [ ] Charts have labeled axes and units, the B=1 baseline is visible, and rejected percentages are annotated where above 0.

**Kickoff prompt:**
> You are Agent C. Read CLAUDE.md, docs/design.md §10, and section C of docs/test-plan.md. Build the Go load generator (open-loop Poisson by default, plus closed-loop) and the Python sweep and plot scripts in `loadgen/` only. Test against your own in-process stub server. Don't import anything from `gateway/`. Launch the gateway only via its documented CLI flags. Do not edit outside `loadgen/`.

---

## Agent D: Docker, compose, Prometheus

- **Owns:** `deploy/`.
- **Inputs:** each component's `make build` command and the build dependencies in CLAUDE.md, plus the port, flag, and metric tables.
- **Outputs:**
  - `gateway.Dockerfile`: multi-stage. A `golang:1.26` builder with **repo root as the build context** (it needs `gen/go`), with `gateway` and `fakeworker` targets, each on a distroless runtime.
  - `worker.Dockerfile`: a `debian:bookworm` builder that installs grpc and protobuf via apt and downloads the ORT release tarball for `TARGETARCH`, then a slim runtime that includes `grpc_health_probe`. The model is **mounted** from `worker/models/`, not baked into the image.
  - `docker-compose.yml`: services `gateway`, `worker` (profile `real`), `fakeworker` (profile `fake`, network alias `worker`), `prometheus`, and an optional `loadgen` service. The gateway uses `--worker-addr=worker:50052`.
  - `prometheus/prometheus.yml` with a 1s scrape of `gateway:2112`, plus `prometheus/queries.md` with ready-to-paste PromQL for p50, p99, batch size, and reject rate.
  - `smoke_test.sh` and a `Makefile` (`build`, `test`, `up-fake`, `up-real`, `down`).
- **Stubs:** until A and B merge, Dockerfiles are written against the documented commands. Validate with `docker compose config` and the `fake` profile once A merges.
- **Acceptance criteria:**
  - [ ] Every section-D check passes after A merges (fake). The real stack passes after B merges.
  - [ ] `docker compose --profile fake up` gives a stack that serves requests, with Prometheus showing the gateway target `up`.
  - [ ] Nothing under `deploy/` hardcodes values that differ from the CLAUDE.md tables.
  - [ ] Images build natively on arm64.

**Kickoff prompt:**
> You are Agent D. Read CLAUDE.md and section D of docs/test-plan.md. Write the Dockerfiles, compose file, Prometheus config, and smoke test in `deploy/` only. Other components may not exist yet in your worktree: build against the commands and flags documented in CLAUDE.md, and validate with `docker compose config`. Use repo root as the build context. Do not edit outside `deploy/`.

---

## Merge order and post-merge work

1. **A** first: it's the hub everything talks to.
2. **C**: rebase, then run `make bench-fake` against the real gateway and fakeworker. This is the first true end-to-end test.
3. **B**: run the gateway against the real worker, rerun the sweep, and check that results are sane.
4. **D**: run the fake and real smoke tests.
5. Back on main (a single session, not a component agent): add the root `README.md` with charts, add a `go.work` (`use ./gen/go ./gateway ./loadgen`) for editor support, and run the final benchmarks natively.

## Integration risks (expect these when merging)

| # | Risk | Symptom | Mitigation |
|---|---|---|---|
| 1 | **Contract drift**: an agent "fixes" the proto in its worktree | Compile errors or silent field mismatches after merge | Frozen-file rule. Before merging, `git diff main -- proto gen CLAUDE.md docs` must be empty. |
| 2 | **gRPC 4 MiB default message limit** | Large batches fail with `RESOURCE_EXHAUSTED` *from the worker*, which looks like backpressure but isn't | Worker accepts 256 MiB. The gateway sets the client max send size and caps image size. Test `ServerTest.AcceptsLargeBatchMessage`. |
| 3 | **YOLO export has a fixed batch dimension** (forgot `dynamic=True`) | Works with B=1 but fails for any batch > 1, and only with the real worker | The worker checks the shape at startup. The export script hardcodes the flag. |
| 4 | **YOLO output layout or version differences** (`[N,84,8400]` vs. transposed) | Garbage boxes, or zero detections | ★ test with bus.jpg. Log the output shape at startup. |
| 5 | **Coordinate bugs** (letterbox undo, RGB/BGR, CHW order) | Boxes shifted or scaled, low scores | Synthetic postprocess tests, plus `BatchMatchesSingleRuns` and `MixedSizesInOneBatch`. |
| 6 | **Fake vs. real performance gap** | Gateway defaults tuned on the fake (e.g. F=2) behave badly on the real CPU worker due to thread oversubscription | F=1 default. Tune only on real-worker sweeps. See the fake-worker.md caveats. |
| 7 | **Flag, port, or metric name mismatch** among gateway, sweep.py, compose, and Prometheus | Sweep launches the gateway with unknown flags. Empty Grafana/PromQL queries. | Single table in CLAUDE.md. `test_sweep_builds_gateway_command` and `TestMetrics_EndpointServesAllNames`. |
| 8 | **Startup ordering**: the worker takes seconds to load the model | First requests get `UNAVAILABLE`, polluting benchmark results | Worker health gates compose (`service_healthy`). sweep.py waits for readiness. Loadgen warmup excludes the early window. |
| 9 | **Protoc/libprotobuf version mismatch** (anaconda protoc 3.20 on PATH vs. Homebrew libprotobuf) | C++ compile errors in `*.pb.h` ("This file was generated by an older version of protoc") | CMake uses the protoc from the protobuf package it links against. Documented in CLAUDE.md. |
| 10 | **Go module paths in Docker**: the `replace ../gen/go` directive needs the repo root as build context | `go build` can't find `batchserve/gen/go` inside the image | Root build context, copying `gen/go` before `go mod download`. |
| 11 | **Benchmark measurement noise**: loadgen, gateway, and worker share the same CPU cores | Throughput limited by the load generator itself, and latency jitter | Run natively, note core counts in the README, limit worker `--intra-op-threads` to leave cores for the gateway and loadgen, and record CPU usage. |
| 12 | **Docker on macOS VM limits** | Numbers inside compose are much worse than native | Headline numbers come from native runs. Compose is only for the demo and Prometheus screenshots. |
| 13 | **Deadline semantics across hops** | Loadgen deadline is shorter than queue wait plus inference, so everything expires and the charts show only errors | Report expired counts per sweep point. Choose `--deadline` (e.g. 1s) above the worst-case W plus inference. |
| 14 | **Shared-file merge conflicts** (root Makefile, .gitignore) | Conflicts on every merge | Only step 0 and post-merge sessions touch root files. Agents add their own `.gitignore` inside their directory. |
