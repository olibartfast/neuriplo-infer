#pragma once

#include "neuriplo/tasks/core/task_interface.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <type_traits>

namespace neuriplo_infer::validation {
namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace neuriplo_tasks;

inline void writeBytes(const fs::path &path, const uint8_t *data, size_t size) {
  std::ofstream stream(path, std::ios::binary);
  if (size != 0) {
    stream.write(reinterpret_cast<const char *>(data),
                 static_cast<std::streamsize>(size));
  }
  stream.close();
  if (!stream) {
    throw std::runtime_error("Could not record tensor: " + path.string());
  }
}

inline json box(const BoundingBox &value) {
  return {value.x, value.y, value.width, value.height};
}

class PredictionRecorder {
public:
  explicit PredictionRecorder(fs::path directory)
      : directory_(std::move(directory)) {
    if (fs::exists(directory_) && !fs::is_empty(directory_)) {
      throw std::runtime_error("Validation output directory must be empty");
    }
    fs::create_directories(directory_);
  }

  json bytes(const std::vector<uint8_t> &data) {
    const auto name = nextFile();
    writeBytes(directory_ / name, data.data(), data.size());
    return {{"file", name}, {"bytes", data.size()}};
  }

  json matrix(const ImageMatrix &value) {
    if (value.empty()) {
      return nullptr;
    }
    const auto name = nextFile();
    writeBytes(directory_ / name, value.data(), value.sizeBytes());
    const char *type = value.pixelType() == PixelType::Float32 ? "FP32"
                       : value.pixelType() == PixelType::Int32 ? "INT32"
                                                               : "UINT8";
    return {{"file", name},
            {"shape", {value.rows(), value.cols(), value.channels()}},
            {"dtype", type},
            {"bytes", value.sizeBytes()}};
  }

  json tensor(const Tensor &value) {
    const auto name = nextFile();
    std::ofstream stream(directory_ / name, std::ios::binary);
    std::string dtype;
    for (const auto &element : value.data) {
      std::visit(
          [&](const auto &number) {
            using T = std::decay_t<decltype(number)>;
            const char *type = std::is_same_v<T, float>     ? "FP32"
                               : std::is_same_v<T, int32_t> ? "INT32"
                               : std::is_same_v<T, int64_t> ? "INT64"
                                                            : "UINT8";
            if (!dtype.empty() && dtype != type) {
              throw std::runtime_error(
                  "Mixed scalar types in one output tensor");
            }
            dtype = type;
            stream.write(reinterpret_cast<const char *>(&number), sizeof(T));
          },
          element);
    }
    stream.close();
    if (!stream) {
      throw std::runtime_error("Could not record output tensor");
    }
    return {{"file", name},
            {"shape", value.shape},
            {"dtype", dtype},
            {"elements", value.data.size()}};
  }

  json result(const Result &value) {
    return std::visit(
        [&](const auto &r) -> json {
          using T = std::decay_t<decltype(r)>;
          if constexpr (std::is_same_v<T, Classification>) {
            return {{"kind", "classification"},
                    {"class_id", r.class_id},
                    {"score", r.class_confidence}};
          } else if constexpr (std::is_same_v<T, Detection>) {
            return {{"kind", "detection"},
                    {"class_id", r.class_id},
                    {"score", r.class_confidence},
                    {"bbox", box(r.bbox)}};
          } else if constexpr (std::is_same_v<T, OpenVocabDetection>) {
            return {{"kind", "open_vocabulary_detection"},
                    {"prompt_index", r.prompt_index},
                    {"label", r.label},
                    {"score", r.score},
                    {"bbox", box(r.bbox)}};
          } else if constexpr (std::is_same_v<T, InstanceSegmentation>) {
            json polygons = json::array();
            for (const auto &polygon : r.polygons) {
              json exterior = json::array();
              for (const auto &point : polygon.exterior) {
                exterior.push_back({point.x, point.y});
              }
              json holes = json::array();
              for (const auto &ring : polygon.holes) {
                json points = json::array();
                for (const auto &point : ring) {
                  points.push_back({point.x, point.y});
                }
                holes.push_back(std::move(points));
              }
              polygons.push_back({{"exterior", exterior}, {"holes", holes}});
            }
            return {{"kind", "instance_segmentation"},
                    {"class_id", r.class_id},
                    {"score", r.class_confidence},
                    {"bbox", box(r.bbox)},
                    {"mask", matrix(r.mask)},
                    {"mask_data", bytes(r.mask_data)},
                    {"mask_shape", {r.mask_height, r.mask_width}},
                    {"polygons", polygons}};
          } else if constexpr (std::is_same_v<T, OpticalFlow>) {
            return {{"kind", "optical_flow"},
                    {"raw_flow", matrix(r.raw_flow)},
                    {"visualization", matrix(r.flow)}};
          } else if constexpr (std::is_same_v<T, VideoClassification>) {
            return {{"kind", "video_classification"},
                    {"class_id", r.class_id},
                    {"score", r.class_confidence},
                    {"label", r.action_label},
                    {"frame_scores", r.frame_scores}};
          } else if constexpr (std::is_same_v<T, PoseEstimation>) {
            json points = json::array();
            for (const auto &point : r.keypoints) {
              points.push_back({{"x", point.x},
                                {"y", point.y},
                                {"confidence", point.confidence},
                                {"visibility", point.visibility}});
            }
            return {{"kind", "pose_estimation"},
                    {"bbox", box(r.bbox)},
                    {"score", r.score},
                    {"keypoints", points}};
          } else if constexpr (std::is_same_v<T, DepthEstimation>) {
            return {{"kind", "depth_estimation"},
                    {"depth", matrix(r.depth)},
                    {"normalized_depth", matrix(r.normalized_depth)},
                    {"min", r.min_depth},
                    {"max", r.max_depth}};
          } else if constexpr (std::is_same_v<T, GaussianSplatting>) {
            json primitives = json::array();
            for (const auto &g : r.gaussians) {
              primitives.push_back({g.x, g.y, g.z, g.opacity, g.scale_x,
                                    g.scale_y, g.scale_z, g.rot_w, g.rot_x,
                                    g.rot_y, g.rot_z, g.sh_r, g.sh_g, g.sh_b});
            }
            return {{"kind", "gaussian_splatting"},
                    {"count", r.num_gaussians},
                    {"primitives", primitives}};
          } else {
            static_assert(std::is_same_v<T, ImageUnderstanding>);
            return {{"kind", "image_understanding"}, {"text", r.text}};
          }
        },
        value);
  }

  void predictions(const Size &size, const std::vector<Result> &results,
                   const std::vector<Tensor> &outputs = {}) {
    json tensors = json::array();
    for (const auto &output : outputs) {
      tensors.push_back(tensor(output));
    }
    json values = json::array();
    for (const auto &prediction : results) {
      values.push_back(result(prediction));
    }
    document["samples"].push_back({{"image_size", {size.width, size.height}},
                                   {"raw_outputs", tensors},
                                   {"predictions", values}});
  }

  void flush() const {
    std::ofstream stream(directory_ / "predictions.json");
    stream << document.dump(2) << '\n';
    stream.close();
    if (!stream) {
      throw std::runtime_error("Could not record predictions");
    }
  }

  json document{{"schema_version", 1},
                {"samples", json::array()},
                {"preprocess_calls", json::array()}};

private:
  std::string nextFile() {
    return "tensor-" + std::to_string(next_++) + ".bin";
  }
  fs::path directory_;
  size_t next_{0};
};

} // namespace neuriplo_infer::validation
