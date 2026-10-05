// Entry point: parse flags, load the model, serve gRPC until SIGINT/SIGTERM.
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "detector.h"
#include "server.h"

namespace {

struct Flags {
  std::string listen = "0.0.0.0:50052";
  worker::DetectorOptions detector;  // model, provider, intra-op-threads, thresholds
};

[[noreturn]] void Usage(const std::string& problem) {
  std::cerr << (problem.empty() ? "" : "error: " + problem + "\n")
            << "usage: worker [--listen=0.0.0.0:50052] [--model=models/yolov8n.onnx]\n"
               "              [--provider=cpu|cuda] [--intra-op-threads=0]\n"
               "              [--conf-threshold=0.25] [--iou-threshold=0.45] [--max-detections=100]\n";
  std::exit(problem.empty() ? 0 : 2);
}

// Only the `--name=value` form is accepted (that is what the CLAUDE.md flag table
// documents). Failing loudly on an unknown or malformed flag beats silently running
// with a default the operator didn't intend.
Flags ParseFlags(int argc, char** argv) {
  Flags flags;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") Usage("");
    const size_t eq = arg.find('=');
    if (arg.rfind("--", 0) != 0 || eq == std::string::npos) Usage("bad argument '" + arg + "'");
    const std::string name = arg.substr(2, eq - 2);
    const std::string value = arg.substr(eq + 1);

    try {
      if (name == "listen") flags.listen = value;
      else if (name == "model") flags.detector.model_path = value;
      else if (name == "provider") flags.detector.provider = value;
      else if (name == "intra-op-threads") flags.detector.intra_op_threads = std::stoi(value);
      else if (name == "conf-threshold") flags.detector.postprocess.conf_threshold = std::stof(value);
      else if (name == "iou-threshold") flags.detector.postprocess.iou_threshold = std::stof(value);
      else if (name == "max-detections") flags.detector.postprocess.max_detections = std::stoi(value);
      else Usage("unknown flag '--" + name + "'");
    } catch (const std::logic_error&) {  // stoi/stof throw invalid_argument or out_of_range
      Usage("bad value for --" + name + ": '" + value + "'");
    }
  }

  const auto& post = flags.detector.postprocess;
  if (flags.detector.intra_op_threads < 0) Usage("--intra-op-threads must be >= 0");
  if (post.conf_threshold < 0 || post.conf_threshold > 1) Usage("--conf-threshold must be in [0, 1]");
  if (post.iou_threshold < 0 || post.iou_threshold > 1) Usage("--iou-threshold must be in [0, 1]");
  if (post.max_detections < 1) Usage("--max-detections must be >= 1");
  return flags;
}

}  // namespace

int main(int argc, char** argv) {
  const Flags flags = ParseFlags(argc, argv);

  // Block SIGINT/SIGTERM in this thread BEFORE any other thread exists. Threads
  // inherit the mask, so ONNX Runtime's and gRPC's worker threads never receive
  // the signals, and the main thread can wait for one with sigwait() below instead of
  // writing an async-signal-handler.
  sigset_t stop_signals;
  sigemptyset(&stop_signals);
  sigaddset(&stop_signals, SIGINT);
  sigaddset(&stop_signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);

  try {
    // 1. Load the model FIRST, and only then open the port. While this runs (it can
    //    take seconds) connections are refused, so a health probe reports "not up".
    //    This throws, and we exit non-zero, on a missing file or a fixed-batch model.
    std::unique_ptr<worker::Detector> detector = worker::CreateOrtDetector(flags.detector);

    // 2. Serve. From here on health reports SERVING.
    worker::WorkerServer server(*detector, flags.listen);
    std::cerr << "[worker] serving on port " << server.port() << " (provider "
              << flags.detector.provider << ")\n";

    // 3. Sleep until asked to stop, then drain: in-flight batches get 10 s to finish.
    int signal_number = 0;
    sigwait(&stop_signals, &signal_number);
    std::cerr << "[worker] signal " << signal_number << ", shutting down\n";
    server.Shutdown(10'000);
    server.Wait();
  } catch (const std::exception& e) {
    std::cerr << "[worker] fatal: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
