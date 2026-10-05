// The gRPC front door: turns an InferBatchRequest into a Detector call and the
// BatchResult back into an InferBatchResponse. No model code lives here.
#pragma once

#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "batchserve/v1/worker.grpc.pb.h"
#include "detector.h"

namespace worker {

// gRPC rejects messages over 4 MiB by default. A batch of 32 JPEGs easily exceeds
// that, and the failure looks exactly like backpressure (RESOURCE_EXHAUSTED coming
// back from the worker), so raise the limit to 256 MiB.
constexpr int kDefaultMaxMessageBytes = 256 * 1024 * 1024;

// Implements the InferenceWorker service on top of a Detector.
class InferenceService final : public batchserve::v1::InferenceWorker::Service {
 public:
  // `detector` is borrowed and must outlive the service.
  explicit InferenceService(Detector& detector) : detector_(detector) {}

  grpc::Status InferBatch(grpc::ServerContext* context,
                          const batchserve::v1::InferBatchRequest* request,
                          batchserve::v1::InferBatchResponse* response) override;

 private:
  Detector& detector_;
};

// A running gRPC server: the InferenceService plus the standard grpc.health.v1
// health service and server reflection (so `grpcurl` works without .proto files).
//
// Health is SERVING as soon as the server exists. That is correct only because a
// WorkerServer needs an already-loaded Detector: main() loads the model first and
// opens the port afterwards, so "port open" implies "model loaded". Before that,
// connections are simply refused, which a health probe also treats as unhealthy.
class WorkerServer {
 public:
  // listen_addr like "0.0.0.0:50052"; use "127.0.0.1:0" to let the OS pick a port
  // (tests do). Throws std::runtime_error if the port can't be bound.
  WorkerServer(Detector& detector, const std::string& listen_addr,
               int max_message_bytes = kDefaultMaxMessageBytes);

  int port() const { return port_; }  // the port actually bound

  // Blocks until the server stops.
  void Wait() { server_->Wait(); }
  // Stops accepting new RPCs and lets in-flight ones finish for up to `grace_ms`.
  void Shutdown(int grace_ms);

 private:
  InferenceService service_;
  std::unique_ptr<grpc::Server> server_;
  int port_ = 0;
};

}  // namespace worker
