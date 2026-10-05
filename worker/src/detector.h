// The detector: encoded images in, detections out. This header deliberately
// includes neither ONNX Runtime nor gRPC, so the server (and its tests) can use a
// fake Detector without needing either.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "postprocess.h"

namespace worker {

// Result for ONE image of a batch.
struct ImageDetections {
  std::string error;                  // non-empty: this image failed (e.g. undecodable)
  std::vector<Detection> detections;  // empty when `error` is set
};

// Time spent in each phase for the whole batch, in microseconds.
struct BatchTiming {
  int64_t preprocess_us = 0;   // decode + letterbox + pack into one tensor
  int64_t inference_us = 0;    // Session::Run only
  int64_t postprocess_us = 0;  // decode output + NMS + undo letterbox
};

struct BatchResult {
  std::vector<ImageDetections> images;  // same length and order as the input
  BatchTiming timing;
};

// Something that can run detection on a batch. There are two implementations: the
// real ONNX Runtime one (CreateOrtDetector) and a stub in the server tests. That
// second implementation is why this is an interface.
class Detector {
 public:
  virtual ~Detector() = default;

  // Runs detection on every image. A bad image gets its own `error` and does NOT
  // fail the others. Throws std::exception only when the WHOLE batch failed
  // (for example ONNX Runtime threw inside Run()).
  // Must be safe to call from several threads at once: gRPC handles each RPC on its own thread.
  virtual BatchResult DetectBatch(const std::vector<std::string_view>& encoded_images) = 0;
};

struct DetectorOptions {
  std::string model_path = "models/yolov8n.onnx";
  std::string provider = "cpu";  // "cpu" or "cuda"
  int intra_op_threads = 0;      // 0 = let ONNX Runtime use every core
  PostprocessParams postprocess;
};

// Loads the model and checks it is usable. Throws std::runtime_error with a
// readable message if the file is missing, or the model has a fixed batch
// dimension, or its output is not the expected [N, 84, anchors] layout.
std::unique_ptr<Detector> CreateOrtDetector(const DetectorOptions& options);

}  // namespace worker
