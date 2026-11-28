#include "TaskRouting.hpp"
#include "utils.hpp"

#include <stdexcept>
#include <string>

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

  if (startsWith(normalized, "lightglue") ||
      startsWith(normalized, "superglue") ||
      startsWith(normalized, "matchanything") ||
      startsWith(normalized, "flexglue")) {
    return neuriplo_tasks::TaskType::KeypointMatching;
  }

  // Video classification models (temporal, multi-frame). Develop adds
  // pevideo/vjepa/videoprism prefix families (neuriplo-tasks@fd94cab
  // VideoClassification rule); the windowed app path is family-generic.
  if (normalized == "timesformer" || normalized == "videomae" ||
      normalized == "vivit" || startsWith(normalized, "pevideo") ||
      startsWith(normalized, "vjepa") || startsWith(normalized, "videoprism")) {
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
  // Depth estimation: Depth-Anything-V2 plus the YOLO26 depth family added in
  // neuriplo-tasks v0.8.0 (any YOLO-prefixed model type containing `depth`).
  // Mirrors TaskFactory's two depth rules exactly; a broader match routed
  // other "*depth*" names to a task TaskFactory would not build for them.
  // Runs after the seg/pose branches so their precedence is unchanged.
  // Develop adds Depth Anything 3 relative-depth models (neuriplo-tasks@fd94cab
  // DepthAnything3 rule: contains "depthanything3" or starts with "da3").
  if ((startsWith(normalized, "yolo") && contains(normalized, "depth")) ||
      contains(normalized, "depthanythingv2") ||
      contains(normalized, "depthanything3") || startsWith(normalized, "da3")) {
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
  // Gaze estimation and monocular 3D detection exist in neuriplo-tasks
  // develop (neuriplo-tasks@fd94cab TaskFactory); app support is gated on
  // NEURIPLO_TASKS_HAS_GAZE_MONO3D so pins whose headers lack the TaskType
  // enumerators keep building. Without the macro these fail fast naming the
  // family instead of silently misrouting to Detection. Throws
  // std::invalid_argument, which every getTaskTypeForModel caller converts
  // to exit 1 + failure report.
#ifdef NEURIPLO_TASKS_HAS_GAZE_MONO3D
  if (normalized == "gaze" || startsWith(normalized, "ethxgaze") ||
      startsWith(normalized, "gazetr") || contains(normalized, "gaze")) {
    return neuriplo_tasks::TaskType::GazeEstimation;
  }
  if (startsWith(normalized, "urbanomni") ||
      startsWith(normalized, "monocular3d") || normalized == "mono3d") {
    return neuriplo_tasks::TaskType::Monocular3DDetection;
  }
#else
  if (normalized == "gaze" || startsWith(normalized, "ethxgaze") ||
      startsWith(normalized, "gazetr") || contains(normalized, "gaze")) {
    throw std::invalid_argument(
        "gaze estimation (--type='" + model_type +
        "') is not supported by this build: the task exists in "
        "neuriplo-tasks develop but app pipeline and rendering support is "
        "not implemented yet");
  }
  if (startsWith(normalized, "urbanomni") ||
      startsWith(normalized, "monocular3d") || normalized == "mono3d") {
    throw std::invalid_argument(
        "monocular 3D detection (--type='" + model_type +
        "') is not supported by this build: the task exists in "
        "neuriplo-tasks develop but app pipeline and rendering support is "
        "not implemented yet");
  }
#endif
  return neuriplo_tasks::TaskType::Detection; // Default for YOLO, RTDETR, etc.
}
