#include "detector.h"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "image.h"

namespace worker {

namespace {

using Clock = std::chrono::steady_clock;

int64_t MicrosSince(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}

// "[-1, 3, 640, 640]" for logs and error messages. ONNX Runtime reports a dynamic
// (unspecified) dimension as -1.
std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) out << (i ? ", " : "") << shape[i];
  out << "]";
  return out.str();
}

class OrtDetector : public Detector {
 public:
  explicit OrtDetector(const DetectorOptions& options)
      : options_(options),
        // The Env owns ONNX Runtime's global state and must outlive the session,
        // which is why it is declared (and so constructed) first.
        env_(ORT_LOGGING_LEVEL_WARNING, "batchserve-worker"),
        session_(nullptr) {
    Ort::SessionOptions session_options;
    if (options.intra_op_threads > 0) {
      // Threads ONE Run() call may use. Leave some cores free when the gateway and
      // load generator share this machine, or the benchmark measures contention.
      session_options.SetIntraOpNumThreads(options.intra_op_threads);
    }
    AppendProvider(session_options, options.provider);

    // Throws Ort::Exception (a std::exception) if the file is missing or invalid.
    session_ = Ort::Session(env_, options.model_path.c_str(), session_options);

    CheckModelShapes();
  }

  BatchResult DetectBatch(const std::vector<std::string_view>& encoded_images) override {
    BatchResult result;
    result.images.resize(encoded_images.size());

    // --- 1. Preprocess: decode, letterbox, pack ---------------------------------
    auto t0 = Clock::now();
    // Images that fail to decode are left OUT of the tensor, so the tensor holds
    // only the good ones. ok_index[k] remembers which request position the k-th
    // tensor slot belongs to, so results go back to the right place.
    std::vector<Image> decoded;
    std::vector<size_t> ok_index;
    for (size_t i = 0; i < encoded_images.size(); ++i) {
      Image img;
      if (DecodeImage(encoded_images[i], &img, &result.images[i].error)) {
        decoded.push_back(std::move(img));
        ok_index.push_back(i);
      }
    }
    if (decoded.empty()) {
      // Nothing to run (empty batch, or every image was bad). Skip the model.
      result.timing.preprocess_us = MicrosSince(t0);
      return result;
    }
    std::vector<LetterboxInfo> letterboxes;
    std::vector<float> input = PreprocessBatch(decoded, &letterboxes);  // [n, 3, 640, 640]
    result.timing.preprocess_us = MicrosSince(t0);

    // --- 2. One Run() for the whole batch ---------------------------------------
    t0 = Clock::now();
    const int64_t n = static_cast<int64_t>(decoded.size());
    const std::vector<int64_t> input_shape = {n, 3, kInputSize, kInputSize};
    // Wrap our vector as a tensor WITHOUT copying. `input` must stay alive until
    // Run() returns, which it does.
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory, input.data(), input.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    std::vector<Ort::Value> outputs = session_.Run(Ort::RunOptions{nullptr}, input_names,
                                                   &input_tensor, 1, output_names, 1);
    result.timing.inference_us = MicrosSince(t0);

    // --- 3. Postprocess each image ----------------------------------------------
    t0 = Clock::now();
    // Output: [n, 84, anchors] (anchors is 8400 for a 640x640 input). Read the real
    // shape instead of assuming, and refuse anything that isn't what we expect.
    const std::vector<int64_t> out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3 || out_shape[0] != n || out_shape[1] != kNumChannels) {
      throw std::runtime_error("unexpected model output shape " + ShapeToString(out_shape) +
                               ", wanted [" + std::to_string(n) + ", 84, anchors]");
    }
    const int num_candidates = static_cast<int>(out_shape[2]);
    const float* output = outputs[0].GetTensorData<float>();
    const size_t per_image = static_cast<size_t>(kNumChannels) * num_candidates;
    for (size_t k = 0; k < decoded.size(); ++k) {
      // Image k's slice of the batch output starts k * 84 * anchors floats in.
      result.images[ok_index[k]].detections =
          Postprocess(output + k * per_image, num_candidates, letterboxes[k],
                      options_.postprocess);
    }
    result.timing.postprocess_us = MicrosSince(t0);
    return result;
  }

 private:
  // session_options is unused in a build without CUDA, hence maybe_unused.
  static void AppendProvider([[maybe_unused]] Ort::SessionOptions& session_options,
                             const std::string& provider) {
    if (provider == "cpu") return;  // ONNX Runtime's default
    if (provider == "cuda") {
#ifdef WITH_CUDA
      // UNTESTED: written from the ONNX Runtime docs, but no GPU was available here.
      OrtCUDAProviderOptions cuda_options;
      session_options.AppendExecutionProvider_CUDA(cuda_options);
      return;
#else
      throw std::runtime_error("--provider=cuda needs a build with -DWITH_CUDA=ON");
#endif
    }
    throw std::runtime_error("unknown --provider '" + provider + "' (use cpu or cuda)");
  }

  // Refuse to start on a model we can't serve correctly, and log the shapes so a
  // wrong export is obvious from the first line of the log.
  void CheckModelShapes() {
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();

    const auto in_shape =
        session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto out_shape =
        session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "[worker] model " << options_.model_path << ": input '" << input_name_ << "' "
              << ShapeToString(in_shape) << ", output '" << output_name_ << "' "
              << ShapeToString(out_shape) << "\n";

    // Input must be [batch, 3, H, W] with a DYNAMIC batch. A model exported
    // without dynamic=True has a literal 1 here: it works for single images and
    // then fails on any real batch, and only with this worker. Fail loudly now.
    if (in_shape.size() != 4 || in_shape[0] != -1) {
      throw std::runtime_error(
          "model input shape is " + ShapeToString(in_shape) +
          ": the batch dimension must be dynamic (-1). Re-export with "
          "`yolo export format=onnx dynamic=True` (scripts/export_model.sh does this).");
    }
    // Output must be [batch, 84, anchors]. Some exports give the transposed
    // [batch, anchors, 84], which would silently produce garbage boxes.
    if (out_shape.size() != 3 || out_shape[1] != kNumChannels) {
      throw std::runtime_error("model output shape is " + ShapeToString(out_shape) +
                               ", expected [batch, 84, anchors] (YOLOv8, 80 COCO classes)");
    }
  }

  DetectorOptions options_;
  Ort::Env env_;
  Ort::Session session_;
  std::string input_name_;
  std::string output_name_;
};

}  // namespace

std::unique_ptr<Detector> CreateOrtDetector(const DetectorOptions& options) {
  return std::make_unique<OrtDetector>(options);
}

}  // namespace worker
