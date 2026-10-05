// Server tests run a real gRPC server on a localhost port the OS picks, with a stub
// Detector standing in for the model. That tests everything the server does (message
// limits, status codes, ordering, health) without needing models/yolov8n.onnx.
#include <gtest/gtest.h>

#include <grpcpp/grpcpp.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "batchserve/v1/worker.grpc.pb.h"
#include "coco_classes.h"
#include "detector.h"
#include "grpc/health/v1/health.grpc.pb.h"
#include "image.h"
#include "server.h"
#include "test_images.h"

using batchserve::v1::InferBatchRequest;
using batchserve::v1::InferBatchResponse;
using batchserve::v1::InferenceWorker;

namespace {

// Stand-in for the model. "Detects" exactly one object per decodable image, with a
// box whose x_max is the image's width, so a test can tell WHICH image a result
// came from. Undecodable images get a per-image error, like the real detector.
class StubDetector : public worker::Detector {
 public:
  worker::BatchResult DetectBatch(const std::vector<std::string_view>& images) override {
    if (fail_whole_batch) throw std::runtime_error("model exploded");
    worker::BatchResult result;
    for (std::string_view bytes : images) {
      worker::ImageDetections out;
      worker::Image img;
      if (!worker::DecodeImage(bytes, &img, &out.error)) {
        result.images.push_back(std::move(out));
        continue;
      }
      worker::Detection d;
      d.box = {1, 2, static_cast<float>(img.width), 4};
      d.class_id = 0;  // "person"
      d.score = 0.9f;
      out.detections.push_back(d);
      result.images.push_back(std::move(out));
    }
    result.timing = {11, 22, 33};
    return result;
  }
  bool fail_whole_batch = false;
};

std::string Jpeg() {  // 16 x 8
  return std::string(reinterpret_cast<const char*>(testimg::kJpeg16x8Red), testimg::kJpeg16x8RedSize);
}
std::string Png() {  // 12 x 6
  return std::string(reinterpret_cast<const char*>(testimg::kPng12x6Green), testimg::kPng12x6GreenSize);
}
std::string GrayPng() {  // 10 x 4
  return std::string(reinterpret_cast<const char*>(testimg::kGrayPng10x4), testimg::kGrayPng10x4Size);
}

void AddImage(InferBatchRequest* req, uint64_t id, const std::string& bytes) {
  auto* img = req->add_images();
  img->set_request_id(id);
  img->set_data(bytes);
}

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    server_ = std::make_unique<worker::WorkerServer>(stub_, "127.0.0.1:0");
    channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(server_->port()),
                                   grpc::InsecureChannelCredentials());
    client_ = InferenceWorker::NewStub(channel_);
  }

  grpc::Status Call(const InferBatchRequest& req, InferBatchResponse* resp) {
    grpc::ClientContext ctx;
    return client_->InferBatch(&ctx, req, resp);
  }

  StubDetector stub_;
  std::unique_ptr<worker::WorkerServer> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<InferenceWorker::Stub> client_;
};

}  // namespace

TEST_F(ServerTest, PreservesOrderAndRequestIds) {
  InferBatchRequest req;
  req.set_batch_id(42);
  // Deliberately NOT sorted, and with images of different widths (16, 12, 10) so we
  // can tell result i really belongs to image i and not just carries the right id.
  AddImage(&req, 7, Jpeg());
  AddImage(&req, 3, Png());
  AddImage(&req, 9, GrayPng());

  InferBatchResponse resp;
  ASSERT_TRUE(Call(req, &resp).ok());
  EXPECT_EQ(resp.batch_id(), 42u);
  ASSERT_EQ(resp.results_size(), 3);
  EXPECT_EQ(resp.results(0).request_id(), 7u);
  EXPECT_EQ(resp.results(1).request_id(), 3u);
  EXPECT_EQ(resp.results(2).request_id(), 9u);
  EXPECT_FLOAT_EQ(resp.results(0).detections(0).box().x_max(), 16);
  EXPECT_FLOAT_EQ(resp.results(1).detections(0).box().x_max(), 12);
  EXPECT_FLOAT_EQ(resp.results(2).detections(0).box().x_max(), 10);
}

TEST_F(ServerTest, FillsDetectionFieldsAndTiming) {
  InferBatchRequest req;
  AddImage(&req, 1, Jpeg());
  InferBatchResponse resp;
  ASSERT_TRUE(Call(req, &resp).ok());
  const auto& det = resp.results(0).detections(0);
  EXPECT_EQ(det.class_id(), 0);
  EXPECT_EQ(det.class_name(), worker::kCocoClasses[0]);  // "person": filled by the worker
  EXPECT_FLOAT_EQ(det.score(), 0.9f);
  EXPECT_FLOAT_EQ(det.box().x_min(), 1);
  EXPECT_FLOAT_EQ(det.box().y_min(), 2);
  EXPECT_FLOAT_EQ(det.box().y_max(), 4);
  EXPECT_EQ(resp.timing().preprocess_us(), 11);
  EXPECT_EQ(resp.timing().inference_us(), 22);
  EXPECT_EQ(resp.timing().postprocess_us(), 33);
}

TEST_F(ServerTest, CorruptImageGetsPerImageError) {
  InferBatchRequest req;
  AddImage(&req, 1, Jpeg());
  AddImage(&req, 2, "this is not an image");
  AddImage(&req, 3, Png());

  InferBatchResponse resp;
  // A bad image must NOT fail the RPC: the other two clients still need their results.
  ASSERT_TRUE(Call(req, &resp).ok());
  ASSERT_EQ(resp.results_size(), 3);
  EXPECT_TRUE(resp.results(0).error().empty());
  EXPECT_EQ(resp.results(0).detections_size(), 1);
  EXPECT_FALSE(resp.results(1).error().empty());
  EXPECT_EQ(resp.results(1).detections_size(), 0);
  EXPECT_EQ(resp.results(1).request_id(), 2u);  // the id is echoed even on error
  EXPECT_TRUE(resp.results(2).error().empty());
  EXPECT_EQ(resp.results(2).detections_size(), 1);
}

TEST_F(ServerTest, EmptyBatchReturnsOk) {
  InferBatchRequest req;
  req.set_batch_id(5);
  InferBatchResponse resp;
  ASSERT_TRUE(Call(req, &resp).ok());
  EXPECT_EQ(resp.results_size(), 0);
  EXPECT_EQ(resp.batch_id(), 5u);
}

TEST_F(ServerTest, AcceptsLargeBatchMessage) {
  // 20 images of ~1 MiB each = ~20 MiB, five times gRPC's 4 MiB default. A JPEG
  // decoder ignores bytes after the end-of-image marker, so padding keeps each one valid.
  const std::string big = Jpeg() + std::string(1024 * 1024, '\0');
  InferBatchRequest req;
  for (int i = 0; i < 20; ++i) AddImage(&req, i, big);
  ASSERT_GT(req.ByteSizeLong(), 20u * 1024 * 1024);

  InferBatchResponse resp;
  grpc::Status status = Call(req, &resp);
  ASSERT_TRUE(status.ok()) << status.error_message();
  ASSERT_EQ(resp.results_size(), 20);
  EXPECT_TRUE(resp.results(19).error().empty());
}

// Control for the test above: with gRPC's default 4 MiB limit the same request is
// rejected. This proves the large-message test really exercises the raised limit.
TEST(ServerLimitTest, FourMiBLimitRejectsLargeBatch) {
  StubDetector stub;
  worker::WorkerServer small_limit(stub, "127.0.0.1:0", 4 * 1024 * 1024);
  auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(small_limit.port()),
                                     grpc::InsecureChannelCredentials());
  auto client = InferenceWorker::NewStub(channel);

  InferBatchRequest req;
  const std::string big = Jpeg() + std::string(1024 * 1024, '\0');
  for (int i = 0; i < 20; ++i) AddImage(&req, i, big);
  InferBatchResponse resp;
  grpc::ClientContext ctx;
  grpc::Status status = client->InferBatch(&ctx, req, &resp);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
}

TEST_F(ServerTest, WholeBatchFailureIsNonOkStatus) {
  stub_.fail_whole_batch = true;
  InferBatchRequest req;
  AddImage(&req, 1, Jpeg());
  InferBatchResponse resp;
  grpc::Status status = Call(req, &resp);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_NE(status.error_message().find("model exploded"), std::string::npos);
}

TEST_F(ServerTest, HealthServingAfterModelLoad) {
  auto health = grpc::health::v1::Health::NewStub(channel_);
  grpc::health::v1::HealthCheckRequest req;  // service "" = the server as a whole
  grpc::health::v1::HealthCheckResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(health->Check(&ctx, req, &resp).ok());
  EXPECT_EQ(resp.status(), grpc::health::v1::HealthCheckResponse::SERVING);

  // Also the specific service name, which is what `grpc_health_probe -service=...` asks for.
  grpc::health::v1::HealthCheckRequest named;
  named.set_service("batchserve.v1.InferenceWorker");
  grpc::ClientContext ctx2;
  ASSERT_TRUE(health->Check(&ctx2, named, &resp).ok());
  EXPECT_EQ(resp.status(), grpc::health::v1::HealthCheckResponse::SERVING);
}
