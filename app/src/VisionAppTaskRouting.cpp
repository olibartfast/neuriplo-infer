#include "TaskRouting.hpp"
#include "utils.hpp"

namespace {
bool startsWith(const std::string &value, const std::string &prefix) {
  return value.size() >= prefix.size() &&
         value.compare(0, prefix.size(), prefix) == 0;
}

bool contains(const std::string &value, const std::string &needle) {
  return value.find(needle) != std::string::npos;
}
} // namespace

vision_core::TaskType getTaskTypeForModel(const std::string &model_type) {
  std::string normalized = normalizeModelType(model_type);

  // Video classification models (temporal, multi-frame)
  if (normalized == "timesformer" || normalized == "videomae" ||
      normalized == "vivit") {
    return vision_core::TaskType::VideoClassification;
  }
  // Single-frame classification models
  if (normalized == "torchvisionclassifier" ||
      normalized == "tensorflowclassifier" || normalized == "vitclassifier" ||
      startsWith(normalized, "resnet") || contains(normalized, "tensorflow")) {
    return vision_core::TaskType::Classification;
  }
  if (contains(normalized, "seg") || normalized == "yoloseg") {
    return vision_core::TaskType::InstanceSegmentation;
  }
  if (normalized == "raft") {
    return vision_core::TaskType::OpticalFlow;
  }
  // Pose estimation: vitpose plus yolo*pose* and EdgeCrafter
  // ecpose*/edgecrafter*pose*, mirroring vision_core::TaskFactory routing.
  if (normalized == "vitpose" || contains(normalized, "pose")) {
    return vision_core::TaskType::PoseEstimation;
  }
  if (contains(normalized, "depthanythingv2")) {
    return vision_core::TaskType::DepthEstimation;
  }
  if (normalized == "lgm" || normalized == "grm" ||
      normalized == "gaussiansplatting" || normalized == "lgmmini" ||
      contains(normalized, "splat")) {
    return vision_core::TaskType::GaussianSplatting;
  }
  if (normalized == "owlv2" || normalized == "owlvit" ||
      normalized == "groundingdino") {
    return vision_core::TaskType::OpenVocabDetection;
  }
  if (normalized == "gemma4" || normalized == "gemma" ||
      normalized == "llama" || normalized == "llamacpp" ||
      normalized == "imageunderstanding") {
    return vision_core::TaskType::ImageUnderstanding;
  }
  return vision_core::TaskType::Detection; // Default for YOLO, RTDETR, etc.
}
