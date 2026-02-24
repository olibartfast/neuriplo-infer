#include "VisionApp.hpp"

#include <cstdlib>
#include <utility>

namespace {

void renderDetectionResults(const std::vector<vision_core::Result>& results, cv::Mat& image,
                            const std::vector<std::string>& classes) {
  for (const auto& result : results) {
    if (std::holds_alternative<vision_core::Detection>(result)) {
      const auto& detection = std::get<vision_core::Detection>(result);
      cv::rectangle(image, detection.bbox, cv::Scalar(255, 0, 0), 3);
      draw_label(image, classes[static_cast<int>(detection.class_id)], detection.class_confidence,
                 detection.bbox.x, detection.bbox.y);
    }
  }
}

void renderClassificationResults(const std::vector<vision_core::Result>& results, cv::Mat& image,
                                 const std::vector<std::string>& classes) {
  std::string result_text = "Classification: ";
  for (const auto& result : results) {
    if (std::holds_alternative<vision_core::Classification>(result)) {
      const auto& classification = std::get<vision_core::Classification>(result);
      if (classification.class_id >= 0 && classification.class_id < classes.size()) {
        result_text += classes[static_cast<int>(classification.class_id)] + " (" +
                       std::to_string(classification.class_confidence) + ")";
      }
    }
  }
  cv::putText(image, result_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 255), 2);
}

void renderVideoClassificationResults(const std::vector<vision_core::Result>& results, cv::Mat& image) {
  std::string result_text = "Action: ";
  for (const auto& result : results) {
    if (std::holds_alternative<vision_core::VideoClassification>(result)) {
      const auto& video_classification = std::get<vision_core::VideoClassification>(result);
      result_text += video_classification.action_label + " (" +
                     std::to_string(video_classification.class_confidence) + ")";
    }
  }
  cv::putText(image, result_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 255), 2);
}

void renderInstanceSegmentationResults(const std::vector<vision_core::Result>& results, cv::Mat& image,
                                       const std::vector<std::string>& classes) {
  for (const auto& result : results) {
    if (std::holds_alternative<vision_core::InstanceSegmentation>(result)) {
      const auto& segmentation = std::get<vision_core::InstanceSegmentation>(result);
      cv::rectangle(image, segmentation.bbox, cv::Scalar(255, 0, 0), 3);
      draw_label(image, classes[static_cast<int>(segmentation.class_id)], segmentation.class_confidence,
                 segmentation.bbox.x, segmentation.bbox.y);

      if (!segmentation.mask.empty()) {
        cv::Mat mask =
          cv::Mat(segmentation.mask_height, segmentation.mask_width, CV_8UC1,
                  const_cast<uint8_t*>(segmentation.mask_data.data()));

        cv::Mat colorMask = cv::Mat::zeros(image.size(), CV_8UC3);
        cv::Scalar color = cv::Scalar(std::rand() & 255, std::rand() & 255, std::rand() & 255);
        colorMask.setTo(color, mask);

        cv::addWeighted(image, 1, colorMask, 0.7, 0, image);
      }
    }
  }
}

void renderOpticalFlowResults(const std::vector<vision_core::Result>& results, cv::Mat& image) {
  for (const auto& result : results) {
    if (std::holds_alternative<vision_core::OpticalFlow>(result)) {
      const auto& flow = std::get<vision_core::OpticalFlow>(result);
      if (!flow.flow.empty()) {
        image = flow.flow.clone();
      }
      std::string flow_text = "Max displacement: " + std::to_string(flow.max_displacement);
      cv::putText(image, flow_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1,
                  cv::Scalar(255, 255, 255), 2);
    }
  }
}

void renderPoseEstimationResults(const std::vector<vision_core::Result>& results, cv::Mat& image,
                                 float confidence_threshold) {
  const std::vector<std::pair<int, int>> skeleton = {
    {0, 1}, {0, 2}, {1, 3},  {2, 4},  // Head
    {5, 6}, {5, 7}, {7, 9},  {6, 8},  {8, 10}, // Arms
    {5, 11}, {6, 12}, {11, 12}, // Torso
    {11, 13}, {13, 15}, {12, 14}, {14, 16} // Legs
  };

  for (const auto& result : results) {
    if (std::holds_alternative<vision_core::PoseEstimation>(result)) {
      const auto& pose = std::get<vision_core::PoseEstimation>(result);

      for (const auto& [i, j] : skeleton) {
        if (i < pose.keypoints.size() && j < pose.keypoints.size()) {
          const auto& kp1 = pose.keypoints[i];
          const auto& kp2 = pose.keypoints[j];
          if (kp1.confidence > confidence_threshold && kp2.confidence > confidence_threshold) {
            cv::line(image, cv::Point(static_cast<int>(kp1.x), static_cast<int>(kp1.y)),
                     cv::Point(static_cast<int>(kp2.x), static_cast<int>(kp2.y)),
                     cv::Scalar(0, 255, 255), 2);
          }
        }
      }

      for (const auto& kp : pose.keypoints) {
        if (kp.confidence > confidence_threshold) {
          cv::circle(image, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)), 5,
                     cv::Scalar(0, 255, 0), -1);
        }
      }

      std::string score_text = "Pose score: " + std::to_string(pose.score);
      cv::putText(image, score_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                  cv::Scalar(0, 255, 255), 2);
    }
  }
}

void renderDepthEstimationResults(const std::vector<vision_core::Result>& results, cv::Mat& image) {
  for (const auto& result : results) {
    if (std::holds_alternative<vision_core::DepthEstimation>(result)) {
      const auto& depth_result = std::get<vision_core::DepthEstimation>(result);

      cv::Mat depth_for_vis;
      if (!depth_result.normalized_depth.empty()) {
        depth_for_vis = depth_result.normalized_depth;
      } else if (!depth_result.depth.empty()) {
        cv::normalize(depth_result.depth, depth_for_vis, 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC1);
      } else {
        continue;
      }

      cv::Mat depth_resized;
      if (depth_for_vis.size() != image.size()) {
        cv::resize(depth_for_vis, depth_resized, image.size(), 0.0, 0.0, cv::INTER_LINEAR);
      } else {
        depth_resized = depth_for_vis;
      }

      cv::Mat depth_u8;
      depth_resized.convertTo(depth_u8, CV_8UC1, 255.0);
      cv::applyColorMap(depth_u8, image, cv::COLORMAP_TURBO);

      std::string depth_text = "Depth min/max: " + std::to_string(depth_result.min_depth) + " / " +
                               std::to_string(depth_result.max_depth);
      cv::putText(image, depth_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                  cv::Scalar(255, 255, 255), 2);
    }
  }
}

} // namespace

void VisionApp::processResults(const std::vector<vision_core::Result>& results, cv::Mat& image) {
  const auto task_type = getTaskType(config.detectorType);

  switch (task_type) {
  case vision_core::TaskType::Detection:
    renderDetectionResults(results, image, classes);
    break;
  case vision_core::TaskType::Classification:
    renderClassificationResults(results, image, classes);
    break;
  case vision_core::TaskType::VideoClassification:
    renderVideoClassificationResults(results, image);
    break;
  case vision_core::TaskType::InstanceSegmentation:
    renderInstanceSegmentationResults(results, image, classes);
    break;
  case vision_core::TaskType::OpticalFlow:
    renderOpticalFlowResults(results, image);
    break;
  case vision_core::TaskType::PoseEstimation:
    renderPoseEstimationResults(results, image, config.confidenceThreshold);
    break;
  case vision_core::TaskType::DepthEstimation:
    renderDepthEstimationResults(results, image);
    break;
  }
}
