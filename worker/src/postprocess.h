// Turning the model's raw output into a short list of detections.
// Pure functions again: numbers in, numbers out, no ONNX Runtime needed.
#pragma once

#include <vector>

#include "image.h"  // LetterboxInfo

namespace worker {

// Axis-aligned box, top-left origin, x grows right and y grows down.
struct Box {
  float x_min = 0, y_min = 0, x_max = 0, y_max = 0;
};

struct Detection {
  Box box;
  int class_id = 0;  // COCO index, 0..79
  float score = 0;   // confidence in [0, 1]
};

// Intersection-over-union of two boxes: overlap area / combined area, in [0, 1].
// 1 means identical, 0 means they don't overlap. Degenerate (zero-area) boxes give 0.
float Iou(const Box& a, const Box& b);

// Class-aware greedy non-maximum suppression.
//
// The model proposes many near-identical boxes for one object. Greedy NMS keeps the
// best-scoring box, drops every remaining box that overlaps it by more than
// `iou_threshold`, then repeats with the next best survivor.
//
// "Class-aware" means a box only suppresses boxes of the SAME class: a person
// standing in front of a bus is two real detections even though their boxes overlap.
//
// Returns the survivors sorted by score, highest first. Ties keep input order.
// Cost is O(n^2) in the worst case, which is fine because the confidence threshold
// has already cut n from 8400 down to a few dozen.
std::vector<Detection> Nms(std::vector<Detection> detections, float iou_threshold);

}  // namespace worker
