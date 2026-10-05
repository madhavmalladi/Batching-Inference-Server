#include "postprocess.h"

#include <algorithm>
#include <limits>

namespace worker {

float Iou(const Box& a, const Box& b) {
  // The overlap rectangle runs from the larger of the two left/top edges to the
  // smaller of the two right/bottom edges. If they don't overlap, one of the
  // sides comes out negative, so clamp it to 0.
  const float inter_w = std::max(0.0f, std::min(a.x_max, b.x_max) - std::max(a.x_min, b.x_min));
  const float inter_h = std::max(0.0f, std::min(a.y_max, b.y_max) - std::max(a.y_min, b.y_min));
  const float inter = inter_w * inter_h;

  const float area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
  const float area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);
  const float uni = area_a + area_b - inter;

  // uni is 0 only when both boxes have zero area; call that "no overlap" rather
  // than dividing 0 by 0 and returning NaN.
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<Detection> Nms(std::vector<Detection> detections, float iou_threshold) {
  // Best first. stable_sort keeps input order among equal scores, so the output is
  // deterministic (std::sort gives no such guarantee).
  std::stable_sort(detections.begin(), detections.end(),
                   [](const Detection& a, const Detection& b) { return a.score > b.score; });

  std::vector<Detection> kept;
  for (const Detection& candidate : detections) {
    // `candidate` survives only if no already-kept box of the same class overlaps
    // it too much. Because we walk in score order, every kept box scores at least
    // as high as the candidate, which is exactly the "keep the best" rule.
    // Boxes dropped earlier are not in `kept`, so they can't suppress anything.
    bool suppressed = false;
    for (const Detection& k : kept) {
      if (k.class_id == candidate.class_id && Iou(k.box, candidate.box) > iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push_back(candidate);
  }
  return kept;
}

std::vector<Detection> Postprocess(const float* output, int num_candidates,
                                   const LetterboxInfo& letterbox,
                                   const PostprocessParams& params) {
  const size_t n = static_cast<size_t>(num_candidates);

  // 1. Best class per candidate. We loop class-by-class (outer) and candidate
  //    (inner) because the data is laid out [channel][candidate]: this way we read
  //    memory front to back instead of jumping 8400 floats at every step.
  std::vector<float> best_score(n, -std::numeric_limits<float>::infinity());
  std::vector<int> best_class(n, 0);
  for (int c = 0; c < kNumClasses; ++c) {
    const float* scores = output + static_cast<size_t>(4 + c) * n;  // [num_candidates]
    for (size_t j = 0; j < n; ++j) {
      if (scores[j] > best_score[j]) {
        best_score[j] = scores[j];
        best_class[j] = c;
      }
    }
  }

  // 2 + 3. Keep confident candidates and convert (cx, cy, w, h) to corners.
  //        Still in 640x640 model space here.
  std::vector<Detection> candidates;
  for (size_t j = 0; j < n; ++j) {
    // Written as !(a > b) rather than (a <= b) so a NaN score is dropped too.
    if (!(best_score[j] > params.conf_threshold)) continue;
    const float cx = output[0 * n + j];
    const float cy = output[1 * n + j];
    const float w = output[2 * n + j];
    const float h = output[3 * n + j];
    Detection d;
    d.box = Box{cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
    d.class_id = best_class[j];
    d.score = best_score[j];
    candidates.push_back(d);
  }

  // 4 + 5. NMS (returns best-first), then cap. Capping AFTER NMS keeps the best
  //        distinct objects; capping before would let duplicates use up the slots.
  std::vector<Detection> kept = Nms(std::move(candidates), params.iou_threshold);
  if (static_cast<int>(kept.size()) > params.max_detections) {
    kept.resize(params.max_detections);
  }

  // 6. Undo the letterbox: subtract the padding, then divide by the scale. (IoU is
  //    unchanged by a shift-and-scale, so NMS above gave the same result it would
  //    have in original-image space.) Then clamp, because the model can predict
  //    boxes that stick out past the image edge.
  const float max_x = static_cast<float>(letterbox.orig_width);
  const float max_y = static_cast<float>(letterbox.orig_height);
  for (Detection& d : kept) {
    d.box.x_min = std::clamp((d.box.x_min - letterbox.pad_x) / letterbox.scale, 0.0f, max_x);
    d.box.x_max = std::clamp((d.box.x_max - letterbox.pad_x) / letterbox.scale, 0.0f, max_x);
    d.box.y_min = std::clamp((d.box.y_min - letterbox.pad_y) / letterbox.scale, 0.0f, max_y);
    d.box.y_max = std::clamp((d.box.y_max - letterbox.pad_y) / letterbox.scale, 0.0f, max_y);
  }
  return kept;
}

}  // namespace worker
