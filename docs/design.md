# Design notes: why it's built this way

These are written to be explained out loud. Each section is a likely interview question.

## 1. Why batch at all?

Running a model has a fixed cost per call: kernel launches, scheduling, weight
reads from memory, thread fork/join. On top of that is a per-image cost. With one
image, the fixed cost dominates and most compute units sit idle. With a batch of
N, the fixed cost is paid once, so

    per-image cost ≈ fixed / N + per_image

Throughput goes up with N until the hardware is saturated. The price is
**latency**: a request may wait for others to arrive, and a bigger batch takes
longer to run. The max_batch_size / max_wait knobs pick a point on that curve.

On CPU (our setup) the gain is smaller than on a GPU but still real. ONNX Runtime
parallelizes one `Run()` across cores, and a batch gives each core more work
per synchronization. Expect roughly 1.5–3× on CPU instead of 5–10× on a GPU, and
say so in the README. The *shape* of the tradeoff is the same.

## 2. Why size-OR-timeout?

- Size only: at low traffic a lone request waits forever.
- Timeout only: at high traffic you send batches bigger than the hardware's sweet
  spot, or you wait even though a full batch is ready.
- Both: max_wait bounds the extra latency batching adds (the worst case is W plus
  inference time), and max_batch_size bounds the work per call.

**Why the timer starts at the oldest request's admission time:** the promise is
"no request waits more than W *in the queue* for batching". If the timer started
when the batcher picked up the request, time spent in a backlog wouldn't count,
and under load requests would wait backlog + W. Counting from admission means a
backlogged batcher dispatches immediately with whatever is queued. Under load,
batches fill up on their own and the timer rarely fires. That's the adaptive
behavior you want.

## 3. Concurrency model (gateway)

```
N handler goroutines ──try-send──▶ [ bounded chan, cap Q ] ──▶ 1 batcher goroutine
        ▲                                                         │ acquire slot (sem, cap F)
        │ result chan (cap 1, one per request)                    ▼
        └──────────────────────────────────────────────── dispatch goroutine per batch
                                                                  │ InferBatch RPC
                                                                  ▼ worker
```

- **One goroutine per request** (grpc-go gives us this for free). It does
  admission, then blocks in `select { result | ctx.Done() }`. Goroutines are
  cheap, so blocking is the simple, readable choice.
- **One batcher goroutine.** Batch formation state (the current batch and its
  deadline) has a single owner, so there are no locks, FIFO order is natural, and
  the logic reads top to bottom. One goroutine is not a bottleneck: forming a
  batch takes microseconds, while inference takes milliseconds.
- **A dispatch goroutine per batch**, capped by a semaphore of size F
  (`max_inflight_batches`). F = 1 is the default because ORT on CPU already uses
  every core for one batch. Two concurrent `Run()` calls would just fight over
  threads. On a GPU, F = 2 lets you overlap preprocessing and network transfer
  with compute.
- **The batcher acquires a slot *before* forming the batch.** If it formed the
  batch first and then waited for a slot, requests arriving during that wait
  couldn't join. Waiting first lets the queue build up, so the next batch is fuller.

## 4. Why a buffered channel as the queue?

A Go buffered channel is already a thread-safe bounded FIFO, and it has the
exact two operations we need:
- `select { case q <- p: ... default: reject }` is a **non-blocking try-enqueue**,
  which is what backpressure needs.
- `<-q` is a blocking dequeue, which is what the batcher needs.

The one thing a channel can't do is remove an item from the middle (e.g. one
whose deadline expired). We use **lazy deletion**: expired items stay in the
queue and are skipped when dequeued. That's correct because expiry only ever
causes skipping. The cost is that an expired request holds a queue slot until
it's dequeued. That's acceptable, and it's the classic tradeoff (timer heaps do
the same thing).

## 5. How results get back to the right client

Each request is a small struct that carries **its own reply channel**:

```go
type pending struct {
    id         uint64
    image      []byte
    ctx        context.Context
    enqueuedAt time.Time
    result     chan result // cap 1
}
```

A batch is `[]*pending`. The worker returns results **in the same order**, so
the dispatcher does `batch[i].result <- res.Results[i]` and also checks that
`request_id`s match, which catches ordering bugs loudly instead of misrouting
silently. No global `map[id]chan` and no mutex are needed. You only need a map
when responses can come back out of order or over a separate stream.

**Why capacity 1:** the dispatcher must never block on a client that has gone
away (deadline expired or cancelled). With a buffer of 1, the send always
succeeds immediately. If nobody reads it, the channel is garbage collected along
with the struct. With an unbuffered channel, that send would block forever:
a goroutine leak that stalls the batch's slot and eventually the whole server.

## 6. Why reject (RESOURCE_EXHAUSTED) instead of queueing without bound

- **Little's law:** L = λ·W, so queue wait ≈ queue length / service rate. If
  arrival rate exceeds service rate, an unbounded queue grows without limit, and
  so does latency.
- Once queue wait exceeds client deadlines, **every** request times out after
  using queue memory and possibly GPU time. Goodput drops to zero even though
  the server is "busy". This is congestion collapse.
- Rejecting fast is *useful information*: the client can back off, retry on
  another replica, or degrade gracefully. A load balancer can route elsewhere. A
  rejection costs microseconds, and a timeout costs the whole deadline.
- A bounded queue also bounds memory: at most Q queued images, plus B×F in flight.
- **Sizing Q:** queue wait ≈ Q / throughput. Pick Q so that wait stays under a
  typical client deadline. For example, 200 img/s throughput with a 1 s deadline
  gives Q ≈ 200. The default of 256 is in that range.

## 7. Deadlines

Deadlines are checked at three points, each for a different reason:
1. **Admission**: don't queue work that is already dead.
2. **Batch formation**: don't spend inference on a request nobody will read.
   This is where most savings come from under overload.
3. **While waiting**: the handler's `select` on `ctx.Done()` answers the client
   at its deadline even if the batch is still running. We do *not* cancel the
   batch, because the other requests in it are still valid.

The worker RPC uses a fixed `--worker-timeout` rather than the batch's minimum
client deadline, so one impatient client can't kill a batch shared with patient
ones. (Another option is the batch's *maximum* deadline. A fixed timeout is
simpler and protects against a hung worker.)

## 8. Proto design choices

- **One image per `Detect` call.** That's the natural client API. Batching is
  the server's job, and clients shouldn't need to coordinate. This mirrors
  Triton's dynamic batcher.
- **Encoded JPEG/PNG bytes, not raw pixels.** Messages are about 10× smaller,
  this is how real clients send images, and decoding happens in the worker, the
  CPU-heavy component, where it can later be parallelized. The gateway never
  parses images, so it stays fast and model-agnostic.
- **Deadline in gRPC metadata, not a field.** gRPC already propagates and
  enforces deadlines across hops, and Go exposes them via `ctx`. A duplicate
  field would be redundant and could disagree with the real deadline.
- **`InferBatch` is unary with repeated images.** It's the simplest thing that
  works: one request, one response, one `Run()`. Streaming would only help if we
  wanted to pipeline partial results, which we don't.
- **Positional results plus `request_id` echo.** Position makes routing O(1) and
  needs no map. The echo is a cheap integrity check across a language boundary.
- **Per-image `error` string.** One corrupt JPEG must not fail 31 healthy
  neighbors. Whole-batch failures use the normal gRPC status.
- **Shared `Detection` type in both services.** The gateway forwards worker
  detections to clients without copying fields between two identical types.
- **`ServerTiming` in the response.** It's diagnostic and costs about 20 bytes.
  The load generator gets batch size and queue wait for every request, so the
  batch-size distribution chart needs no Prometheus scraping.
- **Same proto package for both services.** It's one small system with one
  generated Go package. A real multi-team system would split them so each API
  can be versioned independently.
- **Health uses the standard `grpc.health.v1` service.** grpc_health_probe,
  Kubernetes, and compose healthchecks already speak it, so we don't invent our own RPC.

## 9. Worker design (C++)

Pipeline per batch:
1. **Decode** each image with `stb_image`: one header, no OpenCV, RGB output.
   On failure, set that image's `error` and leave it out of the tensor.
2. **Letterbox** each image to 640×640: scale = min(640/w, 640/h), resize, and
   pad with gray (114) centered. Store (scale, pad_x, pad_y) per image to undo it
   later. The aspect ratio is preserved because stretching distorts objects.
3. **Normalize** to float [0,1], HWC→CHW, and write into one contiguous
   `[N, 3, 640, 640]` tensor at offset i·3·640·640.
4. **`Session::Run`** once. YOLOv8 output is `[N, 84, 8400]`: 8400 candidate
   boxes, each with 4 box values (cx, cy, w, h) and 80 class scores. There is
   no separate objectness score in v8.
5. **Postprocess** per image: take the best class per candidate, apply the
   confidence threshold, convert cxcywh→xyxy, run class-aware greedy NMS (IoU
   0.45), cap at max_detections, undo the letterbox ((x − pad)/scale), and clamp
   to the image bounds.

The model must be exported with a **dynamic batch axis**
(`yolo export model=yolov8n.pt format=onnx dynamic=True opset=17`). The worker
checks the input shape at startup and refuses to start if the batch dimension is fixed.

## 10. Measuring it (loadgen)

- **Open-loop** (fixed arrival rate, Poisson inter-arrivals) is the main mode.
  Real traffic doesn't slow down because your server is slow. Closed-loop
  clients (N workers, each sending again after a response) slow down when the
  server does, which hides queueing latency. This is called *coordinated
  omission*.
- Record every request: send time, latency, status code, and server batch_size
  and queue_wait. Report percentiles over successful requests only, and report
  the rejected and expired rates separately. A server that rejects 50% of
  requests has great p99 latency for the other 50%.
- The baseline is **max_batch_size = 1**. Every chart compares against it.
- Run the load generator, gateway, and worker natively for benchmark numbers.
  Docker on macOS runs inside a VM with capped CPUs. Compose is for
  demonstrating the setup, not for headline numbers.
