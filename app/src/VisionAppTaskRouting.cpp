#include "VisionApp.hpp"

#include <algorithm>
#include <cctype>

namespace {

std::string normalizeModelType(const std::string& model_type) {
  std::string normalized;
  normalized.reserve(model_type.size());

  for (char c : model_type) {
    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      continue;
    }
    if (c == '-' || c == '_') {
      continue;
    }
    normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }

  return normalized;
}

} // namespace

vision_core::TaskType VisionApp::getTaskType(const std::string& model_type) {
  std::string normalized = normalizeModelType(model_type);

  // Video classification models (temporal, multi-frame)
  if (normalized == "timesformer" || normalized == "videomae" || normalized == "vivit") {
    return vision_core::TaskType::VideoClassification;
  }
  // Single-frame classification models
  if (normalized == "torchvisionclassifier" || normalized == "tensorflowclassifier" ||
      normalized == "vitclassifier") {
    return vision_core::TaskType::Classification;
  }
  if (normalized.find("seg") != std::string::npos || normalized == "yoloseg") {
    return vision_core::TaskType::InstanceSegmentation;
  }
  if (normalized == "raft") {
    return vision_core::TaskType::OpticalFlow;
  }
  if (normalized == "vitpose") {
    return vision_core::TaskType::PoseEstimation;
  }
  if (normalized.find("depthanythingv2") != std::string::npos) {
    return vision_core::TaskType::DepthEstimation;
  }
  return vision_core::TaskType::Detection; // Default for YOLO, RTDETR, etc.
}
