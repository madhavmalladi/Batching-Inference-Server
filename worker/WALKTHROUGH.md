# Worker walkthrough

The worker takes a batch of encoded images over gRPC, runs YOLOv8n **once** on all of
them, and returns detections per image, in order. About 500 lines of code in `src/`
(930 counting the comments, of which there are a lot on purpose).

```
InferBatchRequest ──▶ server.cpp ──▶ detector.cpp ──▶ image.cpp        decode, letterbox, pack
                                          │      └──▶ ONNX Runtime     one Session::Run
                                          └─────────▶ postprocess.cpp   best class, NMS, undo letterbox
```

Everything except `detector.cpp` and `server.cpp` is a pure function, so it is tested
without a model.

## Reading order

Read in the order data flows through a request. The tests next to each file are the
fastest way to see what the code is for.

| # | File | What it does | Tests |
|---|---|---|---|
| 1 | `src/image.h` | Types (`Image`, `LetterboxInfo`) and the contract for decode and letterbox. Start here: the comments define the tensor shapes. | |
| 2 | `src/image.cpp` | `DecodeImage` (stb_image, forced to 3 channels, size-limited). `Letterbox` resizes to fit 640×640 keeping aspect ratio, pads with gray 114, and writes straight into the CHW float tensor. `PreprocessBatch` stacks N of them into `[N,3,640,640]`. | `decode_test.cpp`, `letterbox_test.cpp` |
| 3 | `src/postprocess.h` | `Box`, `Detection`, and the exact layout of the model output. | |
| 4 | `src/postprocess.cpp` | `Iou`, class-aware greedy `Nms`, and `Postprocess`: one image's raw `[84, 8400]` slice becomes a short list of boxes in original-image pixels. | `nms_test.cpp`, `postprocess_test.cpp` |
| 5 | `src/detector.h` | The `Detector` interface (the real one and a test stub are its two implementations), result types, options. No ONNX Runtime or gRPC types leak out of it. | |
| 6 | `src/detector.cpp` | The ONNX Runtime wrapper. Startup checks (fixed batch axis, output layout), then `DetectBatch`: preprocess, one `Run`, postprocess. | `detector_test.cpp` |
| 7 | `src/coco_classes.h` | The 80 class names. Index = `class_id`. | |
| 8 | `src/server.h`, `src/server.cpp` | gRPC `InferBatch`, health, reflection, the 256 MiB message limit. Converts `BatchResult` to protobuf. | `server_test.cpp` |
| 9 | `src/main.cpp` | Flags, startup order, signal handling. | |
| 10 | `CMakeLists.txt` | Finds Homebrew protobuf/gRPC first, generates the stubs from `../proto`, defines the targets. | |

Support files: `scripts/export_model.sh` (builds `models/yolov8n.onnx` with a dynamic
batch axis), `scripts/fetch_testdata.sh` (downloads `bus.jpg` and `zidane.jpg`),
`tests/fixtures/static_batch.onnx` (147-byte model with a fixed batch axis, used to
prove the worker rejects it), `tests/proto/.../health.proto` (health *client* stubs
for tests; the worker itself doesn't use it).

Build and run:

```
scripts/export_model.sh && scripts/fetch_testdata.sh   # once; enables the model tests
make test                                              # 44 tests
./build/worker --listen=127.0.0.1:50052
```

Without the model, six tests report `Skipped` and everything else still runs.

## The three decisions a reader is most likely to get wrong

### 1. Results are matched to requests by position, not by tensor slot

In `DetectBatch` (`detector.cpp`), an image that fails to decode is **left out of the
tensor**. So if a batch is `[good, corrupt, good]`, the tensor holds 2 images, and
output slice `k=1` belongs to request `2`, not request `1`. `ok_index[k]` records that
mapping, and results are written to `result.images[ok_index[k]]`.

The tempting simplification is to loop `i` over the output and write to `images[i]`.
That passes every test with clean input and misroutes results exactly when one image
is bad, which is the situation the contract's per-image `error` exists for. The test
that pins this down puts the bad image in the *middle*
(`DetectorTest.BadImageDoesNotFailBatchOrShiftNeighbours`). Putting it last would not
catch the bug.

The same position rule is why `server.cpp` builds `results[i]` from `images[i]` in a
plain loop: the gateway has no map from id to client, it relies on order. The
`request_id` echo is only an integrity check.

### 2. The model output is `[N, 84, 8400]`, channel-major, with no objectness score

`output[n][c][j]` is channel `c` of candidate `j`. In flat memory that is
`n*84*8400 + c*8400 + j`, so **all 8400 center-x values come first**, then all 8400
center-y values, and so on. It is *not* 8400 rows of 84 numbers. Channels 0–3 are
`cx, cy, w, h` in 640×640 model pixels (center and size, not corners), and channels 4–83
are the 80 class scores. YOLOv8 has no objectness channel (v5 did), so the best class
score *is* the confidence.

Three things follow, all in `postprocess.cpp`:
- The argmax loops over classes in the outer loop and candidates in the inner one,
  because that reads memory sequentially. Swapping them is correct but strided.
- Image `k` of the batch starts at `k * 84 * anchors` (`detector.cpp`). The anchor
  count is read from the output tensor's real shape, not hardcoded.
- A *transposed* export (`[N, 8400, 84]`) would run without errors and produce garbage
  boxes. The detector refuses to start unless dimension 1 is 84.

Getting the boxes back to the original image is the other half: subtract the padding,
divide by the scale, then clamp to the **original** width and height (not 640). The
padding stored in `LetterboxInfo` is the integer that was actually used, so the undo is
exact. `UndoesLetterbox` and `ClampsToOriginalSizeNotModelSize` cover both.

### 3. "Health is SERVING" is true by construction, not tracked

There is no NOT_SERVING phase. `main.cpp` loads the model **first**, and only then
constructs `WorkerServer`, which opens the port and sets SERVING. While the model
loads, the port isn't open at all, so a health probe gets "connection refused" and
treats the worker as unhealthy. The contract says "SERVING only after the model has
loaded", and ordering the startup achieves that without a state machine.

The consequence: **don't move the model load after the server starts** (to get a
faster-opening port, say) without also adding real NOT_SERVING handling, or the worker
will report healthy and fail every request. The same goes for a failed load: it exits
non-zero before the port ever opens, which is what lets compose's `service_healthy`
gate the gateway.

One detail that looks redundant but isn't: `server.cpp` calls `SetServingStatus` both
for the whole server and for `batchserve.v1.InferenceWorker`. gRPC's default health
service only knows service names it is told about, and a probe asking for our service
by name got NOT_FOUND until I registered it.

## Smaller traps

- **256 MiB message limit.** gRPC's default is 4 MiB. Exceeding it returns
  `RESOURCE_EXHAUSTED` *from the worker*, which looks like gateway backpressure. A
  control test (`ServerLimitTest`) shows the default really does reject a 20 MiB batch.
- **protoc comes from CMake targets, never from `PATH`.** `/opt/anaconda3/bin/protoc`
  is 3.20 and Homebrew's libprotobuf is 36.x. `CMakeLists.txt` puts the Homebrew prefix
  first and uses `$<TARGET_FILE:protobuf::protoc>`. The configure output prints which
  protobuf it found: check it says `/opt/homebrew`.
- **Truncated JPEGs decode "successfully".** stb_image returns the rows it got and
  fills the rest with gray. Corrupt *headers* are rejected; we chose not to be stricter
  than the decoder. Noted in `image.h` and `decode_test.cpp`.
- **No lock around `Session::Run`.** It is thread-safe, so concurrent batches would
  both run. The gateway's default of one in-flight batch is what prevents two `Run`
  calls from fighting over the same cores.
- **CUDA is written but untested** (`-DWITH_CUDA=ON`, `--provider=cuda`): there was no
  GPU to try it on.
- **Don't compare against `yolo predict` and expect identical boxes.** For a dynamic
  ONNX model, `yolo predict` letterboxes to the *smallest stride-32 rectangle* (it fed
  `bus.jpg` to the model as 640×480, with no padding). We always pad to a full 640×640
  square, because every image in a batch must share one tensor shape. Both are valid,
  but they give different boxes: on `bus.jpg` the bus `x_min` differs by about 12 px
  and scores by up to 0.03. Compare like for like instead: feed Ultralytics'
  `LetterBox(auto=False)` output through ONNX Runtime with its own NMS and
  `scale_boxes`. Done that way (checked on `bus.jpg` and `zidane.jpg`), this worker
  matches to within 0.4 px and 0.001 in score. The preprocessed tensor itself differs
  from Ultralytics' by at most 1/255 (cv2 rounds its resize to 8 bits; we keep floats).
