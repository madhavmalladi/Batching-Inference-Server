// Detector tests. The ones that need the real model skip themselves (instead of
// failing) when models/yolov8n.onnx or the sample images are missing, so CI
// without the model still passes. Run scripts/export_model.sh and
// scripts/fetch_testdata.sh to enable them.
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "coco_classes.h"
#include "detector.h"
#include "test_images.h"

using worker::BatchResult;
using worker::Detection;
using worker::Detector;
using worker::DetectorOptions;

namespace {

const std::string kRoot = WORKER_SOURCE_DIR;
const std::string kModel = kRoot + "/models/yolov8n.onnx";
const std::string kStaticModel = kRoot + "/tests/fixtures/static_batch.onnx";

constexpr int kPerson = 0;
constexpr int kBus = 5;

// Returns the file's bytes, or "" if it doesn't exist.
std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int CountClass(const std::vector<Detection>& dets, int class_id, float min_score) {
  return static_cast<int>(std::count_if(dets.begin(), dets.end(), [&](const Detection& d) {
    return d.class_id == class_id && d.score > min_score;
  }));
}

// Put at the top of a test that needs the real model: skips it (rather than failing)
// when the model or sample images haven't been fetched. A macro, because GTEST_SKIP()
// must run inside the test body itself to end that test.
#define REQUIRE_MODEL()                                                               \
  if (!detector_) {                                                                   \
    GTEST_SKIP() << "needs models/yolov8n.onnx and testdata/*.jpg: run "              \
                    "worker/scripts/export_model.sh and worker/scripts/fetch_testdata.sh"; \
  }

class DetectorTest : public ::testing::Test {
 protected:
  // Loading the model takes a moment, so load it once for the whole suite.
  static void SetUpTestSuite() {
    bus_ = ReadFile(kRoot + "/testdata/bus.jpg");
    zidane_ = ReadFile(kRoot + "/testdata/zidane.jpg");
    if (ReadFile(kModel).empty() || bus_.empty() || zidane_.empty()) return;
    DetectorOptions options;
    options.model_path = kModel;
    detector_ = worker::CreateOrtDetector(options);
  }
  static void TearDownTestSuite() { detector_.reset(); }

  static std::unique_ptr<Detector> detector_;
  static std::string bus_;     // 810x1080 (portrait)
  static std::string zidane_;  // 1280x720 (landscape)
};

std::unique_ptr<Detector> DetectorTest::detector_;
std::string DetectorTest::bus_;
std::string DetectorTest::zidane_;

}  // namespace

// Needs no real model: the fixture is a 147-byte ONNX file made for this test.
TEST_F(DetectorTest, RejectsFixedBatchModel) {
  DetectorOptions options;
  options.model_path = kStaticModel;
  try {
    worker::CreateOrtDetector(options);
    FAIL() << "expected a model with a fixed batch dimension to be rejected";
  } catch (const std::runtime_error& e) {
    // The message must tell the operator what is wrong AND how to fix it.
    EXPECT_NE(std::string(e.what()).find("dynamic"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("[1, 3, 640, 640]"), std::string::npos) << e.what();
  }
}

TEST_F(DetectorTest, MissingModelFileThrows) {
  DetectorOptions options;
  options.model_path = kRoot + "/models/does_not_exist.onnx";
  EXPECT_ANY_THROW(worker::CreateOrtDetector(options));
}

TEST_F(DetectorTest, UnknownProviderThrows) {
  DetectorOptions options;
  options.model_path = kStaticModel;
  options.provider = "tpu";
  EXPECT_THROW(worker::CreateOrtDetector(options), std::runtime_error);
}

TEST_F(DetectorTest, BusImageFindsPersonAndBus) {
  REQUIRE_MODEL();
  BatchResult r = detector_->DetectBatch({bus_});
  ASSERT_EQ(r.images.size(), 1u);
  ASSERT_TRUE(r.images[0].error.empty()) << r.images[0].error;
  const auto& dets = r.images[0].detections;
  EXPECT_GE(CountClass(dets, kBus, 0.5f), 1);
  EXPECT_GE(CountClass(dets, kPerson, 0.5f), 3);
  EXPECT_GT(r.timing.inference_us, 0);
  // The class table and the model must agree about what index 5 means.
  EXPECT_STREQ(worker::kCocoClasses[kBus], "bus");
}

// THE key batching test: a batch must give the same answers as running each image
// alone. If padding, indexing, or output slicing were wrong, this is where it shows.
TEST_F(DetectorTest, BatchMatchesSingleRuns) {
  REQUIRE_MODEL();
  // Alternating two different images catches both "batch mixes up images" and
  // "batch order is wrong" bugs.
  const std::vector<std::string_view> images = {bus_, zidane_, zidane_, bus_};
  BatchResult batch = detector_->DetectBatch(images);
  ASSERT_EQ(batch.images.size(), 4u);

  for (size_t i = 0; i < images.size(); ++i) {
    BatchResult single = detector_->DetectBatch({images[i]});
    const auto& b = batch.images[i].detections;
    const auto& s = single.images[0].detections;
    ASSERT_EQ(b.size(), s.size()) << "image " << i;
    for (size_t k = 0; k < b.size(); ++k) {
      EXPECT_EQ(b[k].class_id, s[k].class_id) << "image " << i << " det " << k;
      EXPECT_NEAR(b[k].score, s[k].score, 1e-3) << "image " << i << " det " << k;
      EXPECT_NEAR(b[k].box.x_min, s[k].box.x_min, 1.0) << "image " << i << " det " << k;
      EXPECT_NEAR(b[k].box.y_min, s[k].box.y_min, 1.0) << "image " << i << " det " << k;
      EXPECT_NEAR(b[k].box.x_max, s[k].box.x_max, 1.0) << "image " << i << " det " << k;
      EXPECT_NEAR(b[k].box.y_max, s[k].box.y_max, 1.0) << "image " << i << " det " << k;
    }
  }
}

TEST_F(DetectorTest, MixedSizesInOneBatch) {
  REQUIRE_MODEL();
  // Three very different shapes in one tensor: 810x1080 portrait, 1280x720
  // landscape, and a 16x8 sliver that gets enlarged 40x.
  const std::string tiny(reinterpret_cast<const char*>(testimg::kJpeg16x8Red),
                         testimg::kJpeg16x8RedSize);
  BatchResult r = detector_->DetectBatch({bus_, zidane_, tiny});
  ASSERT_EQ(r.images.size(), 3u);
  const int widths[] = {810, 1280, 16};
  const int heights[] = {1080, 720, 8};

  // Every box must lie inside ITS OWN image. A box un-letterboxed with another
  // image's scale or padding would stick out or be clamped to the wrong edge.
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_TRUE(r.images[i].error.empty()) << r.images[i].error;
    for (const Detection& d : r.images[i].detections) {
      EXPECT_GE(d.box.x_min, 0.0f);
      EXPECT_GE(d.box.y_min, 0.0f);
      EXPECT_LE(d.box.x_max, static_cast<float>(widths[i]));
      EXPECT_LE(d.box.y_max, static_cast<float>(heights[i]));
    }
  }

  // The bus spans nearly the full width of the portrait frame (reference box is about
  // x 22..805, y 231..756 in bus-image pixels), so its box must be large in THIS
  // image's pixels, not in 640x640 model pixels.
  const Detection* bus = nullptr;
  for (const Detection& d : r.images[0].detections)
    if (d.class_id == kBus && (!bus || d.score > bus->score)) bus = &d;
  ASSERT_NE(bus, nullptr);
  EXPECT_GT(bus->box.x_max - bus->box.x_min, 600.0f);
  EXPECT_GT(bus->box.y_max - bus->box.y_min, 450.0f);

  // zidane.jpg is 1280 wide and the people span it. Coordinates beyond 810 (the
  // bus image's width) prove this image was mapped with its own numbers.
  EXPECT_GE(CountClass(r.images[1].detections, kPerson, 0.5f), 2);
  float max_x = 0;
  for (const Detection& d : r.images[1].detections) max_x = std::max(max_x, d.box.x_max);
  EXPECT_GT(max_x, 900.0f);
}

TEST_F(DetectorTest, BadImageDoesNotFailBatchOrShiftNeighbours) {
  REQUIRE_MODEL();
  // The bad image sits in the MIDDLE: if results were assigned by tensor slot
  // instead of request position, zidane would receive bus's neighbour's results.
  BatchResult r = detector_->DetectBatch({bus_, "not an image", zidane_});
  ASSERT_EQ(r.images.size(), 3u);
  EXPECT_TRUE(r.images[0].error.empty());
  EXPECT_FALSE(r.images[1].error.empty());
  EXPECT_TRUE(r.images[1].detections.empty());
  EXPECT_TRUE(r.images[2].error.empty());
  EXPECT_GE(CountClass(r.images[0].detections, kBus, 0.5f), 1);
  EXPECT_EQ(CountClass(r.images[2].detections, kBus, 0.5f), 0);  // zidane has no bus
  EXPECT_GE(CountClass(r.images[2].detections, kPerson, 0.5f), 2);
}

TEST_F(DetectorTest, AllImagesBadSkipsTheModel) {
  REQUIRE_MODEL();
  BatchResult r = detector_->DetectBatch({"nope", ""});
  ASSERT_EQ(r.images.size(), 2u);
  EXPECT_FALSE(r.images[0].error.empty());
  EXPECT_FALSE(r.images[1].error.empty());
  EXPECT_EQ(r.timing.inference_us, 0);  // Run() was never called
}

TEST_F(DetectorTest, EmptyBatchReturnsNothing) {
  REQUIRE_MODEL();
  BatchResult r = detector_->DetectBatch({});
  EXPECT_TRUE(r.images.empty());
}
