#include <gtest/gtest.h>

#include <string>
#include <string_view>

#include "image.h"
#include "test_images.h"

using worker::DecodeImage;
using worker::Image;

TEST(DecodeTest, ValidJpegDecodesToRgb) {
  Image img;
  std::string err;
  ASSERT_TRUE(DecodeImage({reinterpret_cast<const char*>(testimg::kJpeg16x8Red),
                           testimg::kJpeg16x8RedSize},
                          &img, &err))
      << err;
  EXPECT_EQ(img.width, 16);
  EXPECT_EQ(img.height, 8);
  // 3 channels: the buffer holds exactly W*H*3 bytes.
  ASSERT_EQ(img.rgb.size(), 16u * 8u * 3u);
  // JPEG is lossy, so "red" is only approximately (255,0,0). Check the first pixel
  // is clearly red, which also proves the channel order is R,G,B and not B,G,R.
  EXPECT_GT(img.rgb[0], 200);
  EXPECT_LT(img.rgb[1], 60);
  EXPECT_LT(img.rgb[2], 60);
}

TEST(DecodeTest, PngDecodes) {
  Image img;
  std::string err;
  ASSERT_TRUE(DecodeImage({reinterpret_cast<const char*>(testimg::kPng12x6Green),
                           testimg::kPng12x6GreenSize},
                          &img, &err))
      << err;
  EXPECT_EQ(img.width, 12);
  EXPECT_EQ(img.height, 6);
  ASSERT_EQ(img.rgb.size(), 12u * 6u * 3u);
  // PNG is lossless, so the values are exact.
  EXPECT_EQ(img.rgb[0], 0);
  EXPECT_EQ(img.rgb[1], 255);
  EXPECT_EQ(img.rgb[2], 0);
}

TEST(DecodeTest, GrayscaleBecomesThreeChannels) {
  Image img;
  std::string err;
  ASSERT_TRUE(DecodeImage({reinterpret_cast<const char*>(testimg::kGrayPng10x4),
                           testimg::kGrayPng10x4Size},
                          &img, &err))
      << err;
  EXPECT_EQ(img.width, 10);
  EXPECT_EQ(img.height, 4);
  ASSERT_EQ(img.rgb.size(), 10u * 4u * 3u);
  // The single gray value (200) is copied into R, G and B.
  EXPECT_EQ(img.rgb[0], 200);
  EXPECT_EQ(img.rgb[1], 200);
  EXPECT_EQ(img.rgb[2], 200);
}

TEST(DecodeTest, CorruptBytesReturnError) {
  Image img;
  std::string err;
  EXPECT_FALSE(DecodeImage("this is definitely not an image", &img, &err));
  EXPECT_FALSE(err.empty());

  // A valid file cut off inside its header/metadata must fail cleanly, not crash or
  // read past the end. (We cut a JPEG at 20 bytes and a PNG in half.)
  //
  // NOTE: a JPEG cut off in the middle of its pixel data does NOT fail: stb_image
  // returns the rows it could decode and fills the rest with gray. Like most JPEG
  // decoders it is lenient there, and we deliberately don't add our own check.
  EXPECT_FALSE(DecodeImage({reinterpret_cast<const char*>(testimg::kJpeg16x8Red), 20},
                           &img, &err));
  EXPECT_FALSE(DecodeImage({reinterpret_cast<const char*>(testimg::kPng12x6Green),
                            testimg::kPng12x6GreenSize / 2},
                           &img, &err));

  // Empty input is the other classic edge case.
  EXPECT_FALSE(DecodeImage("", &img, &err));
}
