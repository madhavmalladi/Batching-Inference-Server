# Test plan

Test names are fixed: agents may add tests but must not rename these. Times in
gateway tests are **virtual time** (`testing/synctest`), so "at t=10ms" is exact
and the test runs instantly. Unless stated otherwise, the gateway batcher tests
use the in-process fake (`internal/fakeworker`) behind the `Inferer` interface.

## A. Gateway (`gateway/`, Go, `go test -race ./...`)

### Batching policy (`internal/batcher`)

| Test | Verifies |
|---|---|
| `TestBatcher_SingleRequestTimesOutAlone` | B=8, W=10ms, one request at t=0. Dispatched at exactly t=10ms as a batch of 1 with reason TIMEOUT. |
| `TestBatcher_BurstFillsBatchImmediately` | B=8, W=10ms, 8 requests at t=0. Dispatched at t=0 (no wait) with reason FULL. |
| `TestBatcher_BurstLargerThanBatchSplits` | B=8, W=10ms, F=3, 20 requests at t=0. Batches are 8 and 8 at t=0 (FULL), then 4 at t=10ms (TIMEOUT, measured from those requests' admission). |
| `TestBatcher_TimerStartsAtOldestRequest` | W=10ms. Request 1 at t=0, request 2 at t=8ms. One batch {1,2} at t=10ms, not at t=18ms. |
| `TestBatcher_BacklogDispatchesWithoutWaiting` | F=1, worker blocked. 3 requests queue up, then the worker is released at t=50ms. The next batch (size 3) dispatches at t=50ms with no extra W wait, because the oldest request is already past its deadline. |
| `TestBatcher_MaxWaitZeroIsGreedy` | W=0. Every batch contains exactly what was queued at that moment, and nothing waits. |
| `TestBatcher_MaxBatchSizeOneNeverWaits` | B=1, W=100ms. Every request is dispatched alone and immediately (the baseline). |
| `TestBatcher_PreservesFIFOOrder` | Request IDs within and across batches arrive at the worker in admission order. |
| `TestBatcher_RespectsMaxInflight` | F=2, slow worker, 50 requests. The fake's `MaxConcurrent()` is ≤ 2. |
| `TestBatcher_AcquiresSlotBeforeFormingBatch` | F=1, worker busy until t=20ms. Requests arriving at t=1..15ms all go into the *next* batch together. |

### Admission and backpressure (`internal/server`)

| Test | Verifies |
|---|---|
| `TestDetect_QueueOverflowRejectsWithResourceExhausted` | Q=4, B=1, F=1, worker blocked. 1 request in flight and 4 queued, so the 6th gets `RESOURCE_EXHAUSTED`. |
| `TestDetect_RejectionIsImmediate` | The rejected call returns at the same virtual instant. It never blocks. |
| `TestDetect_RejectedRequestNeverReachesWorker` | After releasing the worker, the worker saw exactly 5 IDs, and the rejected ID isn't among them. |
| `TestDetect_QueueRecoversAfterDrain` | After an overflow and drain, new requests succeed again. |
| `TestDetect_EmptyImageInvalidArgument` | Empty `image` gives `INVALID_ARGUMENT`, and the request isn't queued. |
| `TestDetect_OversizeImageInvalidArgument` | `len(image) > --max-image-bytes` gives `INVALID_ARGUMENT`. |

### Deadlines and cancellation

| Test | Verifies |
|---|---|
| `TestDetect_ExpiredDeadlineRejectedAtAdmission` | A ctx that is already past its deadline gets `DEADLINE_EXCEEDED`. `expired_total{stage="admission"}` is incremented. |
| `TestBatcher_DropsExpiredRequestsBeforeDispatch` | Worker blocked. Request X (5ms deadline) is queued with Y (no deadline). After release at t=20ms, the worker batch is {Y} only. `expired_total{stage="queued"}` is 1. |
| `TestBatcher_AllExpiredBatchIsNotSent` | Every queued request has expired, so the worker gets no call and the slot is released (the next request still works). |
| `TestDetect_DeadlineDuringInferenceReturnsAtDeadline` | Worker takes 100ms and the request's deadline is 30ms. The client gets `DEADLINE_EXCEEDED` at t=30ms, not t=100ms. The other requests in the batch still succeed. |
| `TestDetect_LateResultDoesNotLeakGoroutine` | Same setup. After the batch completes, `synctest.Wait()` returns: no goroutine is blocked sending to the abandoned result channel. |
| `TestDetect_ClientCancelReturnsCanceled` | Cancelling ctx while queued gives `CANCELLED`, and the request is skipped at dispatch. |

### Result routing

| Test | Verifies |
|---|---|
| `TestRouting_EachClientGetsOwnResult` | 200 concurrent requests with distinct payload lengths and B=16. Each response's `box.x_max == len(own payload)`, using the fake's content-derived output. |
| `TestRouting_PerImageErrorFailsOnlyThatRequest` | A batch of 4 where one image is `BADIMG…`. That client gets `INVALID_ARGUMENT` and the other 3 get OK. |
| `TestRouting_WorkerErrorFailsWholeBatch` | The worker returns `UNAVAILABLE`, so every request in the batch gets `UNAVAILABLE`. |
| `TestRouting_ResultCountMismatchIsInternal` | A stub worker returns N−1 results, so all N requests get `INTERNAL`. |
| `TestRouting_RequestIDMismatchIsInternal` | A stub worker returns swapped IDs, so all requests get `INTERNAL`. No silent misrouting. |
| `TestDetect_ResponseIncludesServerTiming` | `batch_size`, `queue_wait_us`, `batch_latency_us`, and `dispatch_reason` match what actually happened. |

### Metrics (`internal/metrics`, prometheus `testutil`)

| Test | Verifies |
|---|---|
| `TestMetrics_BatchSizeHistogram` | Batches of 8, 8, and 4 give count 3 and sum 20. |
| `TestMetrics_BatchesTotalByReason` | `full` and `timeout` labels are counted correctly. |
| `TestMetrics_RejectedCounter` | Queue overflow increments `requests_rejected_total{reason="queue_full"}`. |
| `TestMetrics_QueueDepthGauge` | The gauge reads the number of currently queued requests. |
| `TestMetrics_RequestDurationLabeledByCode` | OK and RESOURCE_EXHAUSTED are recorded in separate series. |
| `TestMetrics_EndpointServesAllNames` | GET `/metrics` contains every metric name from CLAUDE.md. |

### Fake worker (`internal/fakeworker`)

| Test | Verifies |
|---|---|
| `TestFakeWorker_LatencyScalesWithBatchSize` | Defaults: N=1 sleeps 7ms and N=8 sleeps 21ms (virtual time). |
| `TestFakeWorker_DeterministicDetections` | Same bytes give the same output. class_id = crc32 % 80 and x_max = len. |
| `TestFakeWorker_BadImagePerImageError` | `BADIMG` and empty inputs get a per-image error. The others are fine. |
| `TestFakeWorker_FailRate` | fail-rate=1 always fails and 0 never does. A fixed seed is reproducible. |
| `TestFakeWorker_HonorsContextCancel` | A cancelled ctx returns before the sleep ends. |

### Integration (real gRPC on localhost, fakeworker server, `-short` skips the slow ones)

| Test | Verifies |
|---|---|
| `TestIntegration_EndToEndWithFakeWorker` | Gateway and fakeworker over real sockets: 50 concurrent `Detect` calls all succeed with correct routing. |
| `TestIntegration_GatewayStartsBeforeWorker` | Gateway starts first, so requests get `UNAVAILABLE`. The worker starts and requests succeed without a gateway restart. |
| `TestIntegration_BatchingBeatsNoBatching` | Same offered load: throughput with B=8 is greater than 1.5× B=1. Uses real time, so it's skipped with `-short`. |
| `TestIntegration_GracefulShutdown` | SIGTERM-equivalent while requests are in flight: in-flight requests complete, and new ones get `UNAVAILABLE`. |

## B. Worker (`worker/`, GoogleTest, `ctest`)

Tests marked ★ need `models/yolov8n.onnx` and a sample image. They call
`GTEST_SKIP()` with a clear message if those are missing, so CI without the
model still passes.

| Test | Verifies |
|---|---|
| `DecodeTest.ValidJpegDecodesToRgb` | A tiny embedded JPEG decodes to the right w/h with 3 channels. |
| `DecodeTest.PngDecodes` | Same check for PNG. |
| `DecodeTest.GrayscaleBecomesThreeChannels` | A 1-channel input is forced to RGB. |
| `DecodeTest.CorruptBytesReturnError` | Garbage bytes give an error, not a crash. |
| `LetterboxTest.SquareImageNoPadding` | 640×640 input: scale 1 and pad 0. |
| `LetterboxTest.WideImagePadsTopAndBottom` | 1280×640: scale 0.5, pad_y 160, pad_x 0. |
| `LetterboxTest.TallImagePadsLeftAndRight` | The mirror case. |
| `LetterboxTest.PadValueIs114` | Padding pixels equal 114/255 after normalization. |
| `PreprocessTest.BatchTensorShapeAndLayout` | 3 images give a tensor of size 3·3·640·640 in NCHW. Image i's pixel (c,y,x) is at the expected flat index. |
| `PreprocessTest.NormalizesTo01` | 255 becomes 1.0 and 0 becomes 0.0. |
| `PostprocessTest.FiltersBelowConfThreshold` | A synthetic `[1,84,8400]` tensor with one 0.2 candidate gives 0 detections. |
| `PostprocessTest.PicksArgmaxClass` | The class with the highest score is chosen. |
| `PostprocessTest.ConvertsCxCyWhToXyxy` | Known values. |
| `PostprocessTest.UndoesLetterbox` | A box in 640-space maps back to the expected original-image coordinates for a wide image. |
| `PostprocessTest.ClampsToImageBounds` | No coordinate falls outside [0,w]×[0,h]. |
| `PostprocessTest.CapsAtMaxDetections` | 500 candidates give at most 100 detections, highest scores first. |
| `NmsTest.IouOfKnownBoxes` | Identical boxes give 1, disjoint boxes give 0, and the half-overlap case matches the hand-computed value. |
| `NmsTest.SuppressesOverlapSameClass` | Two boxes with IoU 0.8 in the same class: the higher score is kept. |
| `NmsTest.KeepsOverlapDifferentClass` | The same boxes with different classes are both kept (class-aware NMS). |
| `NmsTest.EmptyInput` | No crash and empty output. |
| ★ `DetectorTest.RejectsFixedBatchModel` | Loading a model with a static batch dimension fails with a clear message. (A fixture model, or skip if none is available.) |
| ★ `DetectorTest.BusImageFindsPersonAndBus` | `bus.jpg` yields at least 1 "bus" and at least 3 "person" detections with score > 0.5. |
| ★ `DetectorTest.BatchMatchesSingleRuns` | 4 images run as one batch match 4 single-image runs (same count and class, boxes within 1px, scores within 1e-3). This is the key correctness test for batching. |
| ★ `DetectorTest.MixedSizesInOneBatch` | Images of different sizes in one batch each get correct coordinates in their own frame. |
| `ServerTest.PreservesOrderAndRequestIds` | InferBatch with IDs [7,3,9] returns results with IDs [7,3,9]. The test uses a stub detector, with gRPC in-process. |
| `ServerTest.CorruptImageGetsPerImageError` | One bad image in a batch of 3: that result has `error` set, and the others have detections. |
| `ServerTest.EmptyBatchReturnsOk` | Zero images gives OK with zero results. |
| `ServerTest.AcceptsLargeBatchMessage` | A request of about 20 MiB (beyond the 4 MiB default) is accepted. |
| `ServerTest.HealthServingAfterModelLoad` | The health check returns SERVING. |

## C. Load generator (`loadgen/`)

### Go (`go test -race ./...`)

| Test | Verifies |
|---|---|
| `TestOpenLoop_PoissonRateIsAccurate` | synctest: rate 200/s for 10s sends 2000 ± 5% requests, and the mean inter-arrival is within 5% of 5ms. |
| `TestOpenLoop_DoesNotWaitForResponses` | Against a stub server that takes 1s per request, sends still happen on schedule. This is the coordinated-omission guard. |
| `TestClosedLoop_BoundsOutstandingRequests` | Concurrency=8 means at most 8 requests are in flight at any time. |
| `TestRecorder_PercentilesOnKnownData` | 1..100ms gives p50 = 50ms and p99 = 99ms. The method is documented (nearest-rank). |
| `TestRecorder_ErrorsExcludedFromLatencyPercentiles` | RESOURCE_EXHAUSTED and DEADLINE_EXCEEDED are counted separately and are not in the latency stats. |
| `TestRecorder_WarmupExcluded` | Requests sent during `--warmup` don't show up in the results. |
| `TestRecorder_CapturesServerTiming` | `batch_size` and `queue_wait_us` from the response are written to the CSV. |
| `TestLoadgen_AppliesPerRequestDeadline` | A stub server sees a deadline of about `--deadline`. |
| `TestLoadgen_WritesCsvAndSummaryJson` | Run against an in-process stub `Detector` server: the files exist and the schema matches. |

### Python (`pytest bench/tests`)

| Test | Verifies |
|---|---|
| `test_sweep_matrix_expansion` | The config expands into the expected (B, W, rate) points. |
| `test_sweep_builds_gateway_command` | The gateway flags in the subprocess argv match the CLAUDE.md flag table (subprocess is mocked). |
| `test_sweep_waits_for_gateway_ready` | Readiness polling times out cleanly if the gateway never comes up. |
| `test_load_summary_parses_fixture` | Reads a fixture `summary.json` and CSV. |
| `test_plot_throughput_latency_creates_png` | Builds a chart from fixture data: the file exists and is non-empty. |
| `test_plot_batch_size_distribution_creates_png` | Same check for the batch-size chart. |

## D. Infrastructure (`deploy/`)

| Test | Verifies |
|---|---|
| `compose_config_valid` | `docker compose config -q` exits 0 for every profile. |
| `images_build` | Gateway, fakeworker, and worker images build on linux/arm64. The worker image also builds on linux/amd64 via buildx (can be optional and slow). |
| `smoke_fake_stack` (`smoke_test.sh`) | `docker compose --profile fake up -d`, wait for health, send 20 requests with the loadgen container or grpcurl (all OK), then the Prometheus API `batchserve_batch_size_count > 0` and target `gateway` is `up`, then `down -v`. |
| `smoke_real_stack` (`smoke_test.sh --real`) | Same with the real worker and the mounted model. Detections for `bus.jpg` are non-empty. |
| `worker_healthcheck_gates_gateway` | The compose healthcheck (grpc_health_probe) keeps the worker unhealthy until the model loads. |
