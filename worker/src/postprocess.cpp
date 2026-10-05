#include "postprocess.h"

#include <algorithm>

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

}  // namespace worker
