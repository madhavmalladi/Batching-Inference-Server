#include "server.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "coco_classes.h"

namespace worker {

namespace pb = batchserve::v1;

grpc::Status InferenceService::InferBatch(grpc::ServerContext*, const pb::InferBatchRequest* request,
                                          pb::InferBatchResponse* response) {
  response->set_batch_id(request->batch_id());

  // Contract: an empty batch is OK with zero results. Answer here so no Detector
  // implementation needs to remember this rule.
  if (request->images_size() == 0) return grpc::Status::OK;

  // Borrow the image bytes straight out of the request: no copies of megabytes of JPEG.
  std::vector<std::string_view> images;
  images.reserve(request->images_size());
  for (const pb::WorkerImage& image : request->images()) images.emplace_back(image.data());

  BatchResult result;
  try {
    result = detector_.DetectBatch(images);
  } catch (const std::exception& e) {
    // The whole batch failed (e.g. ONNX Runtime threw). Per the contract that is a
    // non-OK status; the gateway then fails every request in the batch.
    std::cerr << "[worker] batch " << request->batch_id() << " failed: " << e.what() << "\n";
    return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
  }
  if (result.images.size() != images.size()) {
    return grpc::Status(grpc::StatusCode::INTERNAL, "detector returned the wrong number of results");
  }

  // Build the response IN REQUEST ORDER: results[i] belongs to images[i]. The gateway
  // matches results to clients by position, so this ordering is the whole contract.
  for (size_t i = 0; i < images.size(); ++i) {
    pb::ImageResult* out = response->add_results();
    out->set_request_id(request->images(i).request_id());  // echoed, never interpreted

    const ImageDetections& in = result.images[i];
    if (!in.error.empty()) {
      out->set_error(in.error);  // this image failed; detections stay empty
      continue;
    }
    for (const Detection& d : in.detections) {
      pb::Detection* det = out->add_detections();
      det->set_class_id(d.class_id);
      det->set_class_name(kCocoClasses[d.class_id]);  // class_id is 0..79 by construction
      det->set_score(d.score);
      pb::BoundingBox* box = det->mutable_box();
      box->set_x_min(d.box.x_min);
      box->set_y_min(d.box.y_min);
      box->set_x_max(d.box.x_max);
      box->set_y_max(d.box.y_max);
    }
  }

  pb::WorkerTiming* timing = response->mutable_timing();
  timing->set_preprocess_us(result.timing.preprocess_us);
  timing->set_inference_us(result.timing.inference_us);
  timing->set_postprocess_us(result.timing.postprocess_us);
  return grpc::Status::OK;
}

WorkerServer::WorkerServer(Detector& detector, const std::string& listen_addr, int max_message_bytes)
    : service_(detector) {
  // Both of these must happen BEFORE BuildAndStart.
  grpc::EnableDefaultHealthCheckService(true);               // serves grpc.health.v1.Health
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();  // lets grpcurl list services

  grpc::ServerBuilder builder;
  // The last argument receives the bound port, which matters when listen_addr ends in ":0".
  builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials(), &port_);
  builder.RegisterService(&service_);
  // Send limit too: harmless, and symmetric if a response ever grows.
  builder.SetMaxReceiveMessageSize(max_message_bytes);
  builder.SetMaxSendMessageSize(max_message_bytes);
  server_ = builder.BuildAndStart();
  if (!server_ || port_ == 0) {
    throw std::runtime_error("could not start gRPC server on " + listen_addr);
  }
  // See the class comment: constructing a WorkerServer implies the model is loaded.
  grpc::HealthCheckServiceInterface* health = server_->GetHealthCheckService();
  health->SetServingStatus(true);  // the server as a whole: service name ""
  // gRPC's default health service only knows names it was told about, so a probe
  // asking for our service by name would get NOT_FOUND unless we register it too.
  health->SetServingStatus("batchserve.v1.InferenceWorker", true);
}

void WorkerServer::Shutdown(int grace_ms) {
  // Shutdown(deadline) stops accepting new RPCs immediately and waits for running
  // ones until the deadline, so a rolling restart doesn't drop in-flight batches.
  server_->Shutdown(std::chrono::system_clock::now() + std::chrono::milliseconds(grace_ms));
}

}  // namespace worker
