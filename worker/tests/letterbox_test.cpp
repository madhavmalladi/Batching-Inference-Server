#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include "image.h"

using worker::Image;
using worker::kImageTensorSize;
using worker::kInputSize;
using worker::LetterboxInfo;

namespace {

constexpr float kPad = 114.0f / 255.0f;  // the gray used for padding, normalized

// Builds a w x h image where pixel (x, y) channel c has value fn(x, y, c).
Image MakeImage(int w, int h, const std::function<int(int, int, int)>& fn) {
  Image img;
  img.width = w;
  img.height = h;
  img.rgb.resize(static_cast<size_t>(w) * h * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c)
        img.rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = static_cast<uint8_t>(fn(x, y, c));
  return img;
}

Image SolidImage(int w, int h, int value) {
  return MakeImage(w, h, [value](int, int, int) { return value; });
}

// Index of channel c, row y, column x inside ONE image's [3, 640, 640] slice.
size_t At(int c, int y, int x) {
  return (static_cast<size_t>(c) * kInputSize + y) * kInputSize + x;
}

}  // namespace

TEST(LetterboxTest, SquareImageNoPadding) {
  std::vector<float> dst(kImageTensorSize);
  LetterboxInfo info = worker::Letterbox(SolidImage(640, 640, 10), dst.data());
  EXPECT_FLOAT_EQ(info.scale, 1.0f);
  EXPECT_EQ(info.pad_x, 0);
  EXPECT_EQ(info.pad_y, 0);
  EXPECT_EQ(info.orig_width, 640);
  EXPECT_EQ(info.orig_height, 640);
}

TEST(LetterboxTest, WideImagePadsTopAndBottom) {
  std::vector<float> dst(kImageTensorSize);
  // 1280x640 shrinks by half to 640x320, leaving 320 spare rows: 160 above, 160 below.
  LetterboxInfo info = worker::Letterbox(SolidImage(1280, 640, 10), dst.data());
  EXPECT_FLOAT_EQ(info.scale, 0.5f);
  EXPECT_EQ(info.pad_x, 0);
  EXPECT_EQ(info.pad_y, 160);
}

TEST(LetterboxTest, TallImagePadsLeftAndRight) {
  std::vector<float> dst(kImageTensorSize);
  LetterboxInfo info = worker::Letterbox(SolidImage(640, 1280, 10), dst.data());
  EXPECT_FLOAT_EQ(info.scale, 0.5f);
  EXPECT_EQ(info.pad_x, 160);
  EXPECT_EQ(info.pad_y, 0);
}

TEST(LetterboxTest, PadValueIs114) {
  // Fill dst with garbage first: Letterbox must overwrite every float, because the
  // batch buffer is reused and uninitialized memory would leak into the padding.
  std::vector<float> dst(kImageTensorSize, -1.0f);
  worker::Letterbox(SolidImage(1280, 640, 255), dst.data());
  for (int c = 0; c < 3; ++c) {
    EXPECT_FLOAT_EQ(dst[At(c, 0, 0)], kPad);      // top-left corner: padding
    EXPECT_FLOAT_EQ(dst[At(c, 159, 639)], kPad);  // last padded row at the top
    EXPECT_FLOAT_EQ(dst[At(c, 480, 0)], kPad);    // first padded row at the bottom
    EXPECT_FLOAT_EQ(dst[At(c, 639, 639)], kPad);  // bottom-right corner
    // Rows 160..479 are the image (white = 1.0), so padding stops exactly there.
    EXPECT_FLOAT_EQ(dst[At(c, 160, 0)], 1.0f);
    EXPECT_FLOAT_EQ(dst[At(c, 479, 639)], 1.0f);
  }
}

TEST(LetterboxTest, RoundsOddPaddingDown) {
  // 640x641 is not a clean fit: scale = 640/641, so the height rounds to 640 and
  // the width rounds to 639, leaving 1 spare column. It goes on the right, not
  // the left: pad_x = 1 / 2 = 0 (integer division).
  std::vector<float> dst(kImageTensorSize);
  LetterboxInfo info = worker::Letterbox(SolidImage(640, 641, 10), dst.data());
  EXPECT_EQ(info.pad_x, 0);
  EXPECT_EQ(info.pad_y, 0);
  EXPECT_FLOAT_EQ(dst[At(0, 100, 639)], kPad);  // the one spare column
  EXPECT_FLOAT_EQ(dst[At(0, 100, 638)], 10.0f / 255.0f);
}

TEST(PreprocessTest, BatchTensorShapeAndLayout) {
  // Three 640x640 images (scale 1, so pixels are copied unchanged and we can check
  // exact values). The pixel value encodes where it came from: image i, channel c,
  // row y, column x.
  std::vector<Image> images;
  for (int i = 0; i < 3; ++i) {
    images.push_back(MakeImage(640, 640, [i](int x, int y, int c) {
      return (i * 50 + c * 17 + y * 3 + x * 5) % 256;
    }));
  }
  std::vector<LetterboxInfo> infos;
  std::vector<float> batch = worker::PreprocessBatch(images, &infos);

  // [N=3, 3, 640, 640] flattened.
  ASSERT_EQ(batch.size(), 3u * 3u * 640u * 640u);
  ASSERT_EQ(infos.size(), 3u);

  // Spot-check some pixels. Flat index of (image i, channel c, row y, col x) is
  //   ((i * 3 + c) * 640 + y) * 640 + x
  const struct { int i, c, y, x; } probes[] = {
      {0, 0, 0, 0}, {0, 2, 0, 0}, {1, 0, 10, 20}, {1, 1, 639, 639}, {2, 2, 300, 7}, {2, 0, 0, 639},
  };
  for (const auto& p : probes) {
    size_t flat = ((static_cast<size_t>(p.i) * 3 + p.c) * 640 + p.y) * 640 + p.x;
    int expected = (p.i * 50 + p.c * 17 + p.y * 3 + p.x * 5) % 256;
    EXPECT_FLOAT_EQ(batch[flat], expected / 255.0f)
        << "image " << p.i << " channel " << p.c << " y " << p.y << " x " << p.x;
  }
}

TEST(PreprocessTest, NormalizesTo01) {
  std::vector<Image> images = {SolidImage(640, 640, 255), SolidImage(640, 640, 0)};
  std::vector<LetterboxInfo> infos;
  std::vector<float> batch = worker::PreprocessBatch(images, &infos);
  EXPECT_FLOAT_EQ(batch[0], 1.0f);                     // image 0: white -> 1.0
  EXPECT_FLOAT_EQ(batch[kImageTensorSize], 0.0f);      // image 1: black -> 0.0
  EXPECT_FLOAT_EQ(batch[2 * kImageTensorSize - 1], 0.0f);
}

TEST(PreprocessTest, MixedSizesRecordTheirOwnInfo) {
  std::vector<Image> images = {SolidImage(1280, 640, 0), SolidImage(640, 1280, 0),
                               SolidImage(320, 320, 0)};
  std::vector<LetterboxInfo> infos;
  worker::PreprocessBatch(images, &infos);
  ASSERT_EQ(infos.size(), 3u);
  EXPECT_EQ(infos[0].pad_y, 160);
  EXPECT_EQ(infos[1].pad_x, 160);
  EXPECT_FLOAT_EQ(infos[2].scale, 2.0f);  // small images are enlarged, not just padded
  EXPECT_EQ(infos[2].orig_width, 320);
}
