#include "image.h"

#include <algorithm>
#include <cmath>
#include <memory>

// stb_image is a single-file decoder. Defining STB_IMAGE_IMPLEMENTATION in exactly
// one .cpp file pastes its code here. We only enable JPEG and PNG: that is all the
// contract promises, and every extra format is more untrusted-input parsing code.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

namespace worker {

namespace {

// Refuse absurd sizes before allocating. A tiny PNG can claim to be 60000x60000
// pixels (~10 GB as RGB) and would otherwise be an easy way to crash the worker.
constexpr long long kMaxPixels = 100'000'000;

// stb allocates with malloc, so it has to be released with stbi_image_free.
struct StbiDeleter {
  void operator()(unsigned char* p) const { stbi_image_free(p); }
};

}  // namespace

bool DecodeImage(std::string_view bytes, Image* out, std::string* error) {
  if (bytes.empty()) {
    *error = "empty image";
    return false;
  }
  const auto* data = reinterpret_cast<const unsigned char*>(bytes.data());
  const int len = static_cast<int>(bytes.size());

  // Read only the header first so we can check the size before decoding pixels.
  int w = 0, h = 0, channels_in_file = 0;
  if (!stbi_info_from_memory(data, len, &w, &h, &channels_in_file)) {
    *error = std::string("not a decodable JPEG/PNG: ") + stbi_failure_reason();
    return false;
  }
  if (static_cast<long long>(w) * h > kMaxPixels) {
    *error = "image too large: " + std::to_string(w) + "x" + std::to_string(h);
    return false;
  }

  // The last argument (3) forces RGB output whatever the file contains:
  // grayscale is expanded, and an alpha channel is dropped.
  int ignored = 0;
  std::unique_ptr<unsigned char, StbiDeleter> pixels(
      stbi_load_from_memory(data, len, &w, &h, &ignored, 3));
  if (!pixels) {
    *error = std::string("decode failed: ") + stbi_failure_reason();
    return false;
  }

  out->width = w;
  out->height = h;
  out->rgb.assign(pixels.get(), pixels.get() + static_cast<size_t>(w) * h * 3);
  return true;
}

namespace {

// Gray used to fill the border, as a normalized float (114 / 255). 114 is the
// value YOLO was trained with, so any other gray would slightly hurt accuracy.
constexpr float kPadValue = 114.0f / 255.0f;

// Bilinear resize of `img` to new_w x new_h, written straight into the model
// tensor: each output pixel is normalized to [0,1] and stored in CHW order at
// (pad_x, pad_y) inside the 640x640 plane. Doing the resize and the HWC->CHW
// conversion in one pass avoids allocating an intermediate resized image.
//
//   img.rgb is  [H, W, 3]   (interleaved)
//   dst     is  [3, 640, 640]  (planar)
void ResizeIntoTensor(const Image& img, int new_w, int new_h, int pad_x, int pad_y, float* dst) {
  const int w = img.width;
  const int h = img.height;
  // Source pixels covered by one destination pixel. Using the real ratio w / new_w,
  // rather than 1 / scale, keeps the last column exactly at the image edge even
  // after new_w was rounded to a whole number.
  const float x_ratio = static_cast<float>(w) / new_w;
  const float y_ratio = static_cast<float>(h) / new_h;
  constexpr size_t kPlane = static_cast<size_t>(kInputSize) * kInputSize;

  for (int y = 0; y < new_h; ++y) {
    // "+0.5 ... -0.5" aligns pixel CENTERS (not corners) of the two grids, the same
    // convention as OpenCV's INTER_LINEAR. Without it the result drifts half a
    // pixel. Clamp at 0 so the first row/column doesn't read before the image.
    const float fy = std::max(0.0f, (y + 0.5f) * y_ratio - 0.5f);
    const int y0 = std::min(static_cast<int>(fy), h - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float wy = fy - y0;  // weight of row y1; (1 - wy) is the weight of row y0

    for (int x = 0; x < new_w; ++x) {
      const float fx = std::max(0.0f, (x + 0.5f) * x_ratio - 0.5f);
      const int x0 = std::min(static_cast<int>(fx), w - 1);
      const int x1 = std::min(x0 + 1, w - 1);
      const float wx = fx - x0;

      // The four source pixels around the sample point (3 bytes each).
      const uint8_t* p00 = &img.rgb[(static_cast<size_t>(y0) * w + x0) * 3];
      const uint8_t* p01 = &img.rgb[(static_cast<size_t>(y0) * w + x1) * 3];
      const uint8_t* p10 = &img.rgb[(static_cast<size_t>(y1) * w + x0) * 3];
      const uint8_t* p11 = &img.rgb[(static_cast<size_t>(y1) * w + x1) * 3];

      const size_t out_index = static_cast<size_t>(y + pad_y) * kInputSize + (x + pad_x);
      for (int c = 0; c < 3; ++c) {
        const float top = p00[c] + (p01[c] - p00[c]) * wx;
        const float bottom = p10[c] + (p11[c] - p10[c]) * wx;
        const float value = top + (bottom - top) * wy;
        dst[c * kPlane + out_index] = value / 255.0f;
      }
    }
  }
}

}  // namespace

LetterboxInfo Letterbox(const Image& img, float* dst) {
  // One scale for both axes keeps the aspect ratio. min() picks the axis that
  // hits 640 first, so the whole image fits.
  const float scale = std::min(static_cast<float>(kInputSize) / img.width,
                               static_cast<float>(kInputSize) / img.height);
  // Round to whole pixels, and clamp so float error can never produce 641 or 0.
  const int new_w = std::clamp(static_cast<int>(std::lround(img.width * scale)), 1, kInputSize);
  const int new_h = std::clamp(static_cast<int>(std::lround(img.height * scale)), 1, kInputSize);

  LetterboxInfo info;
  info.scale = scale;
  // Integer division: an odd leftover pixel goes on the right/bottom. We store the
  // pad that was ACTUALLY used, so undoing it later is exact.
  info.pad_x = (kInputSize - new_w) / 2;
  info.pad_y = (kInputSize - new_h) / 2;
  info.orig_width = img.width;
  info.orig_height = img.height;

  // Fill everything with gray first, then draw the image on top. This overwrites
  // all kImageTensorSize floats, so a reused buffer can't leak old data.
  std::fill(dst, dst + kImageTensorSize, kPadValue);
  ResizeIntoTensor(img, new_w, new_h, info.pad_x, info.pad_y, dst);
  return info;
}

std::vector<float> PreprocessBatch(const std::vector<Image>& images,
                                   std::vector<LetterboxInfo>* infos) {
  // [N, 3, 640, 640]: image i is the slice starting at i * kImageTensorSize.
  std::vector<float> batch(images.size() * kImageTensorSize);
  infos->resize(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    (*infos)[i] = Letterbox(images[i], batch.data() + i * kImageTensorSize);
  }
  return batch;
}

}  // namespace worker
