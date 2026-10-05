#include "image.h"

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

}  // namespace worker
