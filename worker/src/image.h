// Image decoding and preprocessing: turn encoded JPEG/PNG bytes into the float
// tensor the model reads. Everything here is a pure function (bytes in, numbers
// out), so each step can be unit-tested without a model.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace worker {

// A decoded image. Always 3 channels (RGB), 8 bits per channel.
struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;  // [H, W, 3], row-major, interleaved: R G B R G B ...
};

// Decodes a JPEG or PNG. Returns false and sets *error if the bytes are not a
// usable image; it never throws or crashes on garbage input, because the bytes
// come straight from the network.
//
// A JPEG truncated in the middle of its pixel data still decodes (the missing rows
// come back gray): the decoder is lenient and we don't second-guess it.
//
// We ask the decoder for exactly 3 channels, so grayscale and RGBA inputs are
// converted to RGB here and nothing later has to care about channel count.
bool DecodeImage(std::string_view bytes, Image* out, std::string* error);

// --- Letterbox + normalize ----------------------------------------------------

// The model takes a fixed square input, 640x640.
constexpr int kInputSize = 640;
// Floats in one image's tensor slice: [3, 640, 640].
constexpr size_t kImageTensorSize = 3 * static_cast<size_t>(kInputSize) * kInputSize;

// Everything needed to map a box from 640x640 model space back to the original
// image. Postprocessing applies the inverse of what Letterbox did.
struct LetterboxInfo {
  float scale = 1.0f;  // model pixels per original pixel (same for x and y)
  int pad_x = 0;       // gray columns added on the left (and the same on the right)
  int pad_y = 0;       // gray rows added on top (and the same on the bottom)
  int orig_width = 0;
  int orig_height = 0;
};

// Resizes `img` to fit inside 640x640 WITHOUT changing its aspect ratio, centers
// it, and fills the rest with gray (114). Stretching instead would distort
// objects, and the model was trained on letterboxed images.
//
// Writes the result as normalized floats in CHW order into `dst`, which must have
// room for kImageTensorSize floats:   dst = [3, 640, 640]   (R plane, G, B)
//   * values are pixel / 255, so they lie in [0, 1]
//   * channel-first (CHW) because that is what the ONNX model expects
LetterboxInfo Letterbox(const Image& img, float* dst);

// Letterboxes every image into one contiguous batch tensor:  [N, 3, 640, 640].
// Image i occupies floats [i * kImageTensorSize, (i + 1) * kImageTensorSize).
// *infos is resized to N, with infos[i] describing image i.
std::vector<float> PreprocessBatch(const std::vector<Image>& images,
                                   std::vector<LetterboxInfo>* infos);

}  // namespace worker
