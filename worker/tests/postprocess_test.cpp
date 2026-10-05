#include <gtest/gtest.h>

#include <vector>

#include "image.h"
#include "postprocess.h"

using worker::Detection;
using worker::kNumChannels;
using worker::LetterboxInfo;
using worker::PostprocessParams;

namespace {

constexpr int kN = 8400;  // candidates per image in the real model

// A synthetic model output for one image: [84, 8400], all zeros to start with.
// Zero scores are below any sane threshold, so unset candidates are ignored.
class FakeOutput {
 public:
  FakeOutput() : data_(static_cast<size_t>(kNumChannels) * kN, 0.0f) {}

  // Fills candidate j: box (cx, cy, w, h) in model pixels plus ONE class score.
  void Set(int j, float cx, float cy, float w, float h, int cls, float score) {
    data_[0 * kN + j] = cx;
    data_[1 * kN + j] = cy;
    data_[2 * kN + j] = w;
    data_[3 * kN + j] = h;
    data_[(4 + cls) * kN + j] = score;
  }
  // Sets an extra class score on candidate j (for the argmax test).
  void SetScore(int j, int cls, float score) { data_[(4 + cls) * kN + j] = score; }

  const float* data() const { return data_.data(); }

 private:
  std::vector<float> data_;
};

// "The image was 640x640, so nothing was scaled or padded."
LetterboxInfo Identity() {
  LetterboxInfo info;
  info.scale = 1.0f;
  info.pad_x = 0;
  info.pad_y = 0;
  info.orig_width = 640;
  info.orig_height = 640;
  return info;
}

std::vector<Detection> RunPost(const FakeOutput& out, const LetterboxInfo& info,
                           PostprocessParams params = {}) {
  return worker::Postprocess(out.data(), kN, info, params);
}

}  // namespace

TEST(PostprocessTest, FiltersBelowConfThreshold) {
  FakeOutput out;
  out.Set(0, 320, 320, 50, 50, /*cls=*/0, /*score=*/0.2f);  // below the default 0.25
  EXPECT_EQ(RunPost(out, Identity()).size(), 0u);

  // The same candidate with a higher score is kept: the filter is doing the work.
  FakeOutput out2;
  out2.Set(0, 320, 320, 50, 50, 0, 0.3f);
  EXPECT_EQ(RunPost(out2, Identity()).size(), 1u);
}

TEST(PostprocessTest, PicksArgmaxClass) {
  FakeOutput out;
  out.Set(0, 320, 320, 50, 50, /*cls=*/3, 0.4f);
  out.SetScore(0, 17, 0.9f);  // the winner
  out.SetScore(0, 5, 0.6f);
  auto dets = RunPost(out, Identity());
  ASSERT_EQ(dets.size(), 1u);  // one candidate gives ONE detection, not one per class
  EXPECT_EQ(dets[0].class_id, 17);
  EXPECT_FLOAT_EQ(dets[0].score, 0.9f);
}

TEST(PostprocessTest, ConvertsCxCyWhToXyxy) {
  FakeOutput out;
  out.Set(0, /*cx=*/100, /*cy=*/200, /*w=*/40, /*h=*/60, 0, 0.9f);
  auto dets = RunPost(out, Identity());
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_FLOAT_EQ(dets[0].box.x_min, 80);    // 100 - 40/2
  EXPECT_FLOAT_EQ(dets[0].box.y_min, 170);   // 200 - 60/2
  EXPECT_FLOAT_EQ(dets[0].box.x_max, 120);   // 100 + 40/2
  EXPECT_FLOAT_EQ(dets[0].box.y_max, 230);   // 200 + 60/2
}

TEST(PostprocessTest, UndoesLetterbox) {
  // A 1280x640 image was shrunk by 0.5 to 640x320 and padded 160 rows top and bottom.
  LetterboxInfo info;
  info.scale = 0.5f;
  info.pad_x = 0;
  info.pad_y = 160;
  info.orig_width = 1280;
  info.orig_height = 640;

  FakeOutput out;
  // Corners in model space: (270, 280) to (370, 360).
  out.Set(0, 320, 320, 100, 80, 0, 0.9f);
  auto dets = RunPost(out, info);
  ASSERT_EQ(dets.size(), 1u);
  // x: (270 - 0) / 0.5 = 540 and (370 - 0) / 0.5 = 740
  // y: (280 - 160) / 0.5 = 240 and (360 - 160) / 0.5 = 400
  EXPECT_FLOAT_EQ(dets[0].box.x_min, 540);
  EXPECT_FLOAT_EQ(dets[0].box.x_max, 740);
  EXPECT_FLOAT_EQ(dets[0].box.y_min, 240);
  EXPECT_FLOAT_EQ(dets[0].box.y_max, 400);
}

TEST(PostprocessTest, UndoesLetterboxWithHorizontalPadding) {
  // The mirror case: 640x1280 shrunk to 320x640, padded 160 columns left and right.
  LetterboxInfo info;
  info.scale = 0.5f;
  info.pad_x = 160;
  info.pad_y = 0;
  info.orig_width = 640;
  info.orig_height = 1280;

  FakeOutput out;
  out.Set(0, 320, 320, 100, 80, 0, 0.9f);  // corners (270, 280) to (370, 360)
  auto dets = RunPost(out, info);
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_FLOAT_EQ(dets[0].box.x_min, 220);  // (270 - 160) / 0.5
  EXPECT_FLOAT_EQ(dets[0].box.x_max, 420);  // (370 - 160) / 0.5
  EXPECT_FLOAT_EQ(dets[0].box.y_min, 560);  // 280 / 0.5
  EXPECT_FLOAT_EQ(dets[0].box.y_max, 720);  // 360 / 0.5
}

TEST(PostprocessTest, ClampsToImageBounds) {
  FakeOutput out;
  out.Set(0, 10, 10, 100, 100, 0, 0.9f);      // sticks out past the top-left corner
  out.Set(1, 630, 630, 100, 100, 1, 0.9f);    // sticks out past the bottom-right corner
  auto dets = RunPost(out, Identity());
  ASSERT_EQ(dets.size(), 2u);
  for (const Detection& d : dets) {
    EXPECT_GE(d.box.x_min, 0.0f);
    EXPECT_GE(d.box.y_min, 0.0f);
    EXPECT_LE(d.box.x_max, 640.0f);
    EXPECT_LE(d.box.y_max, 640.0f);
  }
  // Be specific about the two that were actually clamped.
  const Detection& top_left = dets[0].class_id == 0 ? dets[0] : dets[1];
  const Detection& bottom_right = dets[0].class_id == 0 ? dets[1] : dets[0];
  EXPECT_FLOAT_EQ(top_left.box.x_min, 0.0f);   // was -40
  EXPECT_FLOAT_EQ(bottom_right.box.x_max, 640.0f);  // was 680
}

TEST(PostprocessTest, ClampsToOriginalSizeNotModelSize) {
  // A small 100x50 image upscaled 6.4x to 640x320 (pad_y 160). A box reaching the
  // model-space bottom edge must clamp to 50, the ORIGINAL height, not 640.
  LetterboxInfo info;
  info.scale = 6.4f;
  info.pad_x = 0;
  info.pad_y = 160;
  info.orig_width = 100;
  info.orig_height = 50;

  FakeOutput out;
  out.Set(0, 320, 480, 640, 400, 0, 0.9f);  // corners (0, 280) to (640, 680)
  auto dets = RunPost(out, info);
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_FLOAT_EQ(dets[0].box.x_max, 100.0f);
  EXPECT_FLOAT_EQ(dets[0].box.y_max, 50.0f);
}

TEST(PostprocessTest, CapsAtMaxDetections) {
  FakeOutput out;
  // 500 small boxes on a 25 x 20 grid, 24 px apart, so none of them overlap and NMS
  // removes nothing. Scores rise with j, so the best ones are the highest j.
  for (int j = 0; j < 500; ++j) {
    const float cx = 10.0f + 24.0f * (j % 25);
    const float cy = 10.0f + 24.0f * (j / 25);
    out.Set(j, cx, cy, 10, 10, 0, 0.3f + 0.001f * j);
  }
  PostprocessParams params;
  params.max_detections = 100;
  auto dets = RunPost(out, Identity(), params);
  ASSERT_EQ(dets.size(), 100u);
  // Highest scores first: j = 499 down to j = 400.
  EXPECT_NEAR(dets.front().score, 0.3f + 0.001f * 499, 1e-5);
  EXPECT_NEAR(dets.back().score, 0.3f + 0.001f * 400, 1e-5);
  for (size_t i = 1; i < dets.size(); ++i) EXPECT_GE(dets[i - 1].score, dets[i].score);
}

TEST(PostprocessTest, NmsRemovesDuplicateCandidates) {
  // Two candidates for the same object (IoU ~ 0.9): only the better one survives.
  FakeOutput out;
  out.Set(0, 320, 320, 100, 100, 2, 0.9f);
  out.Set(1, 322, 320, 100, 100, 2, 0.8f);
  auto dets = RunPost(out, Identity());
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_FLOAT_EQ(dets[0].score, 0.9f);
}
