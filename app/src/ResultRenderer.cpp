#include "ResultRenderer.hpp"

#include "AppImageOps.hpp"
#include "neuriplo/tasks/core/vision/image.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace {

namespace imageops = neuriplo_infer::imageops;
using neuriplo_tasks::vision::Image;
using neuriplo_tasks::vision::Point;
using neuriplo_tasks::vision::Rect;

template <typename ResultT, typename Func>
void forEachResultOfType(const std::vector<neuriplo_tasks::Result> &results,
                         Func render) {
  for (const auto &result : results) {
    std::visit(
        [&](const auto &value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, ResultT>) {
            render(value);
          }
        },
        result);
  }
}

std::string formatScore(float value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value;
  return out.str();
}

void renderDetectionResults(const std::vector<neuriplo_tasks::Result> &results,
                            Image &image,
                            const std::vector<std::string> &classes) {
  forEachResultOfType<neuriplo_tasks::Detection>(
      results, [&](const auto &detection) {
        imageops::rectangle(image, detection.bbox, imageops::Color{255, 0, 0},
                            3);
        drawLabelBand(image, detection.bbox.x, detection.bbox.y,
                      labelForClass(detection.class_id, classes) + " " +
                          formatScore(detection.class_confidence),
                      imageops::Color{255, 0, 0});
      });
}

void renderOpenVocabDetectionResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image) {
  forEachResultOfType<neuriplo_tasks::OpenVocabDetection>(
      results, [&](const auto &detection) {
        imageops::rectangle(image, detection.bbox, imageops::Color{0, 165, 255},
                            3);
        drawLabelBand(image, detection.bbox.x, detection.bbox.y,
                      detection.label + " " + formatScore(detection.score),
                      imageops::Color{0, 165, 255});
      });
}

// One tight band per ranked label: each background is sized to its own
// text (via textSize) plus padding, black on yellow. Bands pack from the
// top-left margin with a small gap; overlong labels clip at the image edge.
void renderRankedLabels(const std::vector<std::string> &labels, Image &image) {
  if (labels.empty() || image.empty()) {
    return;
  }
  const int scale = labelTextScale(image);
  const int margin =
      std::min(10 * scale, std::min(image.cols(), image.rows()) / 10);
  const int max_w = std::max(1, image.cols() - 2 * margin);
  const int pad = 2 * scale;
  int top = margin;
  for (const auto &label : labels) {
    const Point text_extent = imageops::textSize(label, scale);
    const int band_w = std::min(max_w, text_extent.x + 2 * pad);
    const int band_h = text_extent.y + 2 * pad;
    imageops::fillRectangle(image, Rect(margin, top, band_w, band_h),
                            imageops::Color{0, 255, 255});
    imageops::drawText(image, label, margin + pad, top + pad,
                       imageops::Color{255, 255, 255}, scale);
    top += band_h + pad * 2;
  }
}

void renderClassificationResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image,
    const std::vector<std::string> &classes) {
  std::vector<std::string> labels;
  forEachResultOfType<neuriplo_tasks::Classification>(
      results, [&](const auto &classification) {
        labels.push_back(labelForClass(classification.class_id, classes) +
                         " (" + formatScore(classification.class_confidence) +
                         ")");
      });
  renderRankedLabels(labels, image);
}

void renderVideoClassificationResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image,
    const std::vector<std::string> &classes) {
  std::vector<std::string> labels;
  forEachResultOfType<neuriplo_tasks::VideoClassification>(
      results, [&](const auto &classification) {
        // neuriplo-tasks does not set action_label; resolve from the labels
        // file the same way still-image classification does.
        const std::string name =
            classification.action_label.empty()
                ? labelForClass(classification.class_id, classes)
                : classification.action_label;
        labels.push_back(name + " (" +
                         formatScore(classification.class_confidence) + ")");
      });
  renderRankedLabels(labels, image);
}

std::vector<Point>
toRenderContour(const std::vector<neuriplo_tasks::vision::Point2f> &ring) {
  std::vector<Point> contour;
  contour.reserve(ring.size());
  for (const auto &point : ring) {
    contour.emplace_back(static_cast<int>(std::lround(point.x)),
                         static_cast<int>(std::lround(point.y)));
  }
  return contour;
}

// Draws the polygon representation produced when
// TaskConfig::segmentation_output is Polygon (neuriplo-tasks v0.7.0): the
// exterior ring and any hole rings are stroked as closed outlines. Nothing is
// filled -- a filled polygon is visually indistinguishable from the mask
// overlay, which defeats the point of asking for the vector representation.
void overlaySegmentationPolygons(
    const std::vector<neuriplo_tasks::SegmentationPolygon> &polygons,
    Image &image) {
  const imageops::Color color{static_cast<std::uint8_t>(std::rand() & 255),
                              static_cast<std::uint8_t>(std::rand() & 255),
                              static_cast<std::uint8_t>(std::rand() & 255)};

  for (const auto &polygon : polygons) {
    if (polygon.exterior.size() < 3) {
      continue;
    }
    const std::vector<std::vector<Point>> exterior{
        toRenderContour(polygon.exterior)};
    imageops::polylines(image, exterior, /*closed=*/true, color, 2);

    // Holes get the same colour but a thinner stroke, so interior boundaries
    // stay distinguishable from the outer perimeter.
    std::vector<std::vector<Point>> holes;
    holes.reserve(polygon.holes.size());
    for (const auto &hole : polygon.holes) {
      if (hole.size() >= 3) {
        holes.push_back(toRenderContour(hole));
      }
    }
    if (!holes.empty()) {
      imageops::polylines(image, holes, /*closed=*/true, color, 1);
    }
  }
}

void renderInstanceSegmentationResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image,
    const std::vector<std::string> &classes) {
  forEachResultOfType<neuriplo_tasks::InstanceSegmentation>(
      results, [&](const auto &segmentation) {
        imageops::rectangle(image, segmentation.bbox,
                            imageops::Color{255, 0, 0}, 3);
        drawLabelBand(image, segmentation.bbox.x, segmentation.bbox.y,
                      labelForClass(segmentation.class_id, classes) + " " +
                          formatScore(segmentation.class_confidence),
                      imageops::Color{255, 0, 0});

        if (!segmentation.polygons.empty()) {
          overlaySegmentationPolygons(segmentation.polygons, image);
        } else if (!segmentation.mask.empty()) {
          Image mask = neuriplo_tasks::toImage(segmentation.mask);
          if (mask.cols() != image.cols() || mask.rows() != image.rows()) {
            mask = imageops::resize(mask, image.cols(), image.rows(),
                                    imageops::Interpolation::Nearest);
          }
          const imageops::Color color{
              static_cast<std::uint8_t>(std::rand() & 255),
              static_cast<std::uint8_t>(std::rand() & 255),
              static_cast<std::uint8_t>(std::rand() & 255)};
          imageops::maskedAlphaBlend(image, color, mask, 0.7);
        }
      });
}

void renderOpticalFlowResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image) {
  forEachResultOfType<neuriplo_tasks::OpticalFlow>(
      results, [&](const auto &flow) {
        if (!flow.flow.empty()) {
          image = neuriplo_tasks::toImage(flow.flow);
        }
      });
}

void renderPoseEstimationResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image,
    float confidence_threshold) {
  const std::vector<std::pair<size_t, size_t>> skeleton = {
      {0, 1},   {0, 2},   {1, 3},   {2, 4},  {5, 6},  {5, 7},
      {7, 9},   {6, 8},   {8, 10},  {5, 11}, {6, 12}, {11, 12},
      {11, 13}, {13, 15}, {12, 14}, {14, 16}};

  forEachResultOfType<neuriplo_tasks::PoseEstimation>(
      results, [&](const auto &pose) {
        for (const auto &[i, j] : skeleton) {
          if (i < pose.keypoints.size() && j < pose.keypoints.size()) {
            const auto &kp1 = pose.keypoints[i];
            const auto &kp2 = pose.keypoints[j];
            if (kp1.confidence > confidence_threshold &&
                kp2.confidence > confidence_threshold) {
              imageops::line(
                  image,
                  Point(static_cast<int>(kp1.x), static_cast<int>(kp1.y)),
                  Point(static_cast<int>(kp2.x), static_cast<int>(kp2.y)),
                  imageops::Color{0, 255, 255}, 2);
            }
          }
        }

        for (const auto &kp : pose.keypoints) {
          if (kp.confidence > confidence_threshold) {
            imageops::fillCircle(
                image, Point(static_cast<int>(kp.x), static_cast<int>(kp.y)), 5,
                imageops::Color{0, 255, 0});
          }
        }
      });
}

void renderDepthEstimationResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image) {
  forEachResultOfType<neuriplo_tasks::DepthEstimation>(
      results, [&](const auto &depth_result) {
        Image depth_for_vis;
        if (!depth_result.normalized_depth.empty()) {
          depth_for_vis =
              neuriplo_tasks::toImage(depth_result.normalized_depth);
        } else if (!depth_result.depth.empty()) {
          depth_for_vis = imageops::normalizeMinMax(
              neuriplo_tasks::toImage(depth_result.depth), 0.0, 1.0);
        } else {
          return;
        }

        if (depth_for_vis.cols() != image.cols() ||
            depth_for_vis.rows() != image.rows()) {
          depth_for_vis =
              imageops::resize(depth_for_vis, image.cols(), image.rows(),
                               imageops::Interpolation::Linear);
        }
        depth_for_vis.convertTo(neuriplo_tasks::vision::PixelType::UInt8,
                                255.0);
        imageops::applyTurboColormap(depth_for_vis, image);
      });
}

void renderImageUnderstandingResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image) {
  forEachResultOfType<neuriplo_tasks::ImageUnderstanding>(
      results, [&](const auto &understanding) {
        std::cout << understanding.text << '\n';
        renderRankedLabels({understanding.text}, image);
      });
}

#ifdef NEURIPLO_TASKS_HAS_GAZE_MONO3D
// Single gaze arrow per result, drawn from the frame center along the
// projected direction (dx = direction[0], dy = direction[1]). The input is a
// normalized face crop, so the center is the only stable anchor: the app has
// no face box. Length is a quarter of the smaller frame side.
void renderGazeEstimationResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image) {
  const int length = std::min(image.cols(), image.rows()) / 4;
  const Point origin{image.cols() / 2, image.rows() / 2};
  forEachResultOfType<neuriplo_tasks::GazeEstimation>(
      results, [&](const auto &gaze) {
        const Point tip{static_cast<int>(
                            std::lround(origin.x + length * gaze.direction[0])),
                        static_cast<int>(std::lround(
                            origin.y + length * gaze.direction[1]))};
        imageops::fillCircle(image, origin, 4, imageops::Color{0, 255, 255});
        imageops::line(image, origin, tip, imageops::Color{0, 255, 255}, 3);
        std::ostringstream readout;
        readout << "yaw " << std::fixed << std::setprecision(2) << gaze.yaw
                << " pitch " << gaze.pitch;
        const std::string text = readout.str();
        const int scale = labelTextScale(image);
        const int pad = 2 * scale;
        const Point size = imageops::textSize(text, scale);
        const int plate_x = origin.x - size.x / 2;
        const int plate_y = origin.y + 10 * scale;
        imageops::fillRectangle(
            image, Rect(plate_x, plate_y, size.x + 2 * pad, size.y + 2 * pad),
            imageops::Color{0, 0, 0});
        imageops::drawText(image, text, plate_x + pad, plate_y + pad,
                           imageops::Color{255, 255, 255}, scale);
      });
}

// Box plus cuboid wireframe: ground quad 0-1-2-3, roof quad 4-5-6-7, four
// verticals i-(i+4), following the canonical ground-first corner layout the
// sibling postprocessor guarantees.
void renderMonocular3DResults(
    const std::vector<neuriplo_tasks::Result> &results, Image &image,
    const std::vector<std::string> &classes) {
  forEachResultOfType<neuriplo_tasks::Monocular3DDetection>(
      results, [&](const auto &detection) {
        imageops::rectangle(image, detection.bbox, imageops::Color{255, 0, 0},
                            3);
        drawLabelBand(image, detection.bbox.x, detection.bbox.y,
                      labelForClass(detection.class_id, classes) + " " +
                          formatScore(detection.class_confidence),
                      imageops::Color{255, 0, 0});
        const auto &corners = detection.projected_corners;
        auto point = [](const neuriplo_tasks::CuboidKeypoint &corner) {
          return Point{static_cast<int>(std::lround(corner.x)),
                       static_cast<int>(std::lround(corner.y))};
        };
        constexpr int kGround[4] = {0, 1, 2, 3};
        for (int i = 0; i < 4; ++i) {
          imageops::line(image, point(corners[kGround[i]]),
                         point(corners[kGround[(i + 1) % 4]]),
                         imageops::Color{0, 255, 255}, 2);
          imageops::line(image, point(corners[kGround[i] + 4]),
                         point(corners[kGround[(i + 1) % 4] + 4]),
                         imageops::Color{0, 255, 255}, 2);
          imageops::line(image, point(corners[kGround[i]]),
                         point(corners[kGround[i] + 4]),
                         imageops::Color{0, 255, 255}, 2);
        }
      });
}
#endif

} // namespace

std::string labelForClass(float class_id,
                          const std::vector<std::string> &classes) {
  const int class_index = static_cast<int>(class_id);
  const std::string fallback = std::to_string(class_index);
  if (class_id < 0.0F) {
    return fallback;
  }

  const auto index = static_cast<size_t>(class_index);
  if (index < classes.size()) {
    return classes[index];
  }
  return fallback;
}

void DefaultResultRenderer::render(
    const std::vector<neuriplo_tasks::Result> &results, Image &image,
    const RenderContext &context) {
  switch (context.task_type) {
  case neuriplo_tasks::TaskType::Detection:
    renderDetectionResults(results, image, context.classes);
    break;
  case neuriplo_tasks::TaskType::Classification:
    renderClassificationResults(results, image, context.classes);
    break;
  case neuriplo_tasks::TaskType::VideoClassification:
    renderVideoClassificationResults(results, image, context.classes);
    break;
  case neuriplo_tasks::TaskType::InstanceSegmentation:
    renderInstanceSegmentationResults(results, image, context.classes);
    break;
  case neuriplo_tasks::TaskType::OpticalFlow:
    renderOpticalFlowResults(results, image);
    break;
  case neuriplo_tasks::TaskType::PoseEstimation:
    renderPoseEstimationResults(results, image, context.confidence_threshold);
    break;
  case neuriplo_tasks::TaskType::DepthEstimation:
    renderDepthEstimationResults(results, image);
    break;
  case neuriplo_tasks::TaskType::OpenVocabDetection:
    renderOpenVocabDetectionResults(results, image);
    break;
  case neuriplo_tasks::TaskType::ImageUnderstanding:
    renderImageUnderstandingResults(results, image);
    break;
#ifdef NEURIPLO_TASKS_HAS_GAZE_MONO3D
  case neuriplo_tasks::TaskType::Monocular3DDetection:
    renderMonocular3DResults(results, image, context.classes);
    break;
  case neuriplo_tasks::TaskType::GazeEstimation:
    renderGazeEstimationResults(results, image);
    break;
#endif
  case neuriplo_tasks::TaskType::GaussianSplatting:
    break;
  case neuriplo_tasks::TaskType::KeypointMatching:
    if (context.pair_source_width <= 0) {
      throw std::invalid_argument(
          "Keypoint matching rendering requires a pair canvas");
    }
    forEachResultOfType<neuriplo_tasks::KeypointMatching>(
        results, [&](const auto &pair) {
          int count = 0;
          for (const auto &match : pair.matches) {
            if (match.score <= context.confidence_threshold)
              continue;
            if (match.source_index < 0 || match.target_index < 0 ||
                static_cast<size_t>(match.source_index) >=
                    pair.source_keypoints.size() ||
                static_cast<size_t>(match.target_index) >=
                    pair.target_keypoints.size()) {
              throw std::invalid_argument(
                  "Keypoint match index is outside the pair");
            }
            const auto &a =
                pair.source_keypoints[static_cast<size_t>(match.source_index)];
            const auto &b =
                pair.target_keypoints[static_cast<size_t>(match.target_index)];
            const Point source(static_cast<int>(std::lround(a.x)),
                               static_cast<int>(std::lround(a.y)));
            const Point target(context.pair_source_width +
                                   static_cast<int>(std::lround(b.x)),
                               static_cast<int>(std::lround(b.y)));
            const imageops::Color color{
                static_cast<std::uint8_t>(64 + count * 71 % 192),
                static_cast<std::uint8_t>(64 + count * 131 % 192),
                static_cast<std::uint8_t>(64 + count * 43 % 192)};
            imageops::line(image, source, target, color, 1);
            imageops::fillCircle(image, source, 3, color);
            imageops::fillCircle(image, target, 3, color);
            ++count;
          }
        });
    break;
  }
}
