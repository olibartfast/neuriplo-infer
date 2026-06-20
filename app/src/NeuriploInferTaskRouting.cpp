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

neuriplo_tasks::TaskType getTaskTypeForModel(const std::string &model_type) {
  std::string normalized = normalizeModelType(model_type);

  // Video classification models (temporal, multi-frame)
  if (normalized == "timesformer" || normalized == "videomae" ||
      normalized == "vivit") {
    return neuriplo_tasks::TaskType::VideoClassification;
  }
  // Single-frame classification models
  if (normalized == "torchvisionclassifier" ||
      normalized == "tensorflowclassifier" || normalized == "vitclassifier" ||
      startsWith(normalized, "resnet") || contains(normalized, "tensorflow")) {
    return neuriplo_tasks::TaskType::Classification;
  }
  if (contains(normalized, "seg") || normalized == "yoloseg") {
    return neuriplo_tasks::TaskType::InstanceSegmentation;
  }
  if (normalized == "raft") {
    return neuriplo_tasks::TaskType::OpticalFlow;
  }
  // Pose estimation: vitpose, yolo*pose*, EdgeCrafter ecpose*, and RF-DETR
  // pose/keypoints. Mirrors neuriplo_tasks::TaskFactory RfDetrPose / YoloPose /
  // EdgeCrafterPose / VitPose routing.
  if (normalized == "vitpose" || contains(normalized, "pose") ||
      (startsWith(normalized, "rfdetr") &&
       (contains(normalized, "keypoint") || contains(normalized, "kpt")))) {
    return neuriplo_tasks::TaskType::PoseEstimation;
  }
  if (contains(normalized, "depthanythingv2")) {
    return neuriplo_tasks::TaskType::DepthEstimation;
  }
  if (normalized == "lgm" || normalized == "grm" ||
      normalized == "gaussiansplatting" || normalized == "lgmmini" ||
      contains(normalized, "splat")) {
    return neuriplo_tasks::TaskType::GaussianSplatting;
  }
  if (normalized == "owlv2" || normalized == "owlvit" ||
      normalized == "groundingdino") {
    return neuriplo_tasks::TaskType::OpenVocabDetection;
  }
  if (normalized == "gemma4" || normalized == "gemma" ||
      normalized == "llama" || normalized == "llamacpp" ||
      normalized == "imageunderstanding") {
    return neuriplo_tasks::TaskType::ImageUnderstanding;
  }
  return neuriplo_tasks::TaskType::Detection; // Default for YOLO, RTDETR, etc.
}
