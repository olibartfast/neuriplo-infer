#include "TaskRouting.hpp"
#include "utils.hpp"

vision_core::TaskType getTaskTypeForModel(const std::string &model_type) {
  std::string normalized = normalizeModelType(model_type);

  // Video classification models (temporal, multi-frame)
  if (normalized == "timesformer" || normalized == "videomae" ||
      normalized == "vivit") {
    return vision_core::TaskType::VideoClassification;
  }
  // Single-frame classification models
  if (normalized == "torchvisionclassifier" ||
      normalized == "tensorflowclassifier" || normalized == "vitclassifier") {
    return vision_core::TaskType::Classification;
  }
  if (normalized.find("seg") != std::string::npos || normalized == "yoloseg") {
    return vision_core::TaskType::InstanceSegmentation;
  }
  if (normalized == "raft") {
    return vision_core::TaskType::OpticalFlow;
  }
  // Pose estimation: vitpose plus yolo*pose* and EdgeCrafter
  // ecpose*/edgecrafter*pose*, mirroring vision_core::TaskFactory routing.
  if (normalized == "vitpose" || normalized.find("pose") != std::string::npos) {
    return vision_core::TaskType::PoseEstimation;
  }
  if (normalized.find("depthanythingv2") != std::string::npos) {
    return vision_core::TaskType::DepthEstimation;
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
