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

}  // namespace worker
