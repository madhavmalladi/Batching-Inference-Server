#include <gtest/gtest.h>

#include <vector>

#include "postprocess.h"

using worker::Box;
using worker::Detection;

namespace {
Detection Det(Box box, int class_id, float score) { return Detection{box, class_id, score}; }
}  // namespace

TEST(NmsTest, IouOfKnownBoxes) {
  const Box a{0, 0, 10, 10};
  EXPECT_FLOAT_EQ(worker::Iou(a, a), 1.0f);                      // identical
  EXPECT_FLOAT_EQ(worker::Iou(a, Box{20, 20, 30, 30}), 0.0f);    // disjoint
  EXPECT_FLOAT_EQ(worker::Iou(a, Box{10, 0, 20, 10}), 0.0f);     // sharing only an edge

  // Half overlap: b is a shifted right by 5. Intersection = 5 x 10 = 50.
  // Union = 100 + 100 - 50 = 150. IoU = 50 / 150 = 1/3.
  EXPECT_NEAR(worker::Iou(a, Box{5, 0, 15, 10}), 1.0f / 3.0f, 1e-6);

  // A box fully inside another: intersection = the small box (25), union = the big one (100).
  EXPECT_FLOAT_EQ(worker::Iou(a, Box{2.5f, 2.5f, 7.5f, 7.5f}), 0.25f);

  // Zero-area boxes must give 0, not NaN from dividing 0 by 0.
  EXPECT_FLOAT_EQ(worker::Iou(Box{5, 5, 5, 5}, Box{5, 5, 5, 5}), 0.0f);
}

TEST(NmsTest, SuppressesOverlapSameClass) {
  // Intersection 100, union 125, so IoU = 0.8, above the 0.45 threshold.
  Box high{0, 0, 10, 10};
  Box low{0, 0, 10, 12.5f};
  ASSERT_NEAR(worker::Iou(high, low), 0.8f, 1e-6);

  // Put the lower score FIRST so the test proves the sort, not the input order.
  auto kept = worker::Nms({Det(low, 7, 0.6f), Det(high, 7, 0.9f)}, 0.45f);
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
}

TEST(NmsTest, KeepsOverlapDifferentClass) {
  Box box{0, 0, 10, 10};
  auto kept = worker::Nms({Det(box, 0, 0.6f), Det(box, 5, 0.9f)}, 0.45f);
  ASSERT_EQ(kept.size(), 2u);
  EXPECT_EQ(kept[0].class_id, 5);  // sorted by score, highest first
  EXPECT_EQ(kept[1].class_id, 0);
}

TEST(NmsTest, EmptyInput) {
  EXPECT_TRUE(worker::Nms({}, 0.45f).empty());
}

TEST(NmsTest, KeepsBoxesBelowThresholdOverlap) {
  // IoU is 1/3, below 0.45, so both survive.
  auto kept = worker::Nms({Det({0, 0, 10, 10}, 1, 0.9f), Det({5, 0, 15, 10}, 1, 0.8f)}, 0.45f);
  EXPECT_EQ(kept.size(), 2u);
}

TEST(NmsTest, SuppressionChainsThroughSurvivorsOnly) {
  // A overlaps B, B overlaps C, but A does not overlap C. Greedy NMS keeps A, drops B
  // (suppressed by A). B is gone, so it can no longer suppress C: A and C both survive.
  Detection a = Det({0, 0, 10, 10}, 1, 0.9f);
  Detection b = Det({4, 0, 14, 10}, 1, 0.8f);   // IoU(a,b) = 60/140 = 0.43, above 0.4
  Detection c = Det({8, 0, 18, 10}, 1, 0.7f);   // IoU(b,c) = 0.43, but IoU(a,c) = 20/180 = 0.11
  auto kept = worker::Nms({a, b, c}, 0.4f);
  ASSERT_EQ(kept.size(), 2u);
  EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
  EXPECT_FLOAT_EQ(kept[1].score, 0.7f);
}
