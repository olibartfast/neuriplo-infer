#pragma once

// Decoder for the ensemble result envelope: turns the fixed tensor set a
// server-side postprocessing ensemble returns back into neuriplo-tasks results.
//
// Works on kserve::InferOutput, whose payloads are raw little-endian bytes.
// Shapes and datatypes come from the platform ensemble contract.
//
// This lives here, not in neuriplo-kserve-client: the client is a pure protocol
// peer with no task types. Interpreting the envelope is the adapter's job.

#include "KserveTypes.hpp"

#include "neuriplo/tasks/core/image_matrix.hpp"
#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/segmentation_types.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace neuriplo_infer {

// Detections past NUM_DETECTIONS are contractual padding.
inline constexpr int kEnvelopeMaxDetections = 100;

enum class EnvelopeVariant { Detection, Mask, Polygon };

inline const kserve::InferOutput *
findEnvelopeTensor(const std::vector<kserve::InferOutput> &outputs,
                   const std::string &name) {
  for (const auto &output : outputs) {
    if (output.name == name) {
      return &output;
    }
  }
  return nullptr;
}

inline const kserve::InferOutput *
requireEnvelopeTensor(const std::vector<kserve::InferOutput> &outputs,
                      const std::string &name, const std::string &datatype) {
  const auto *tensor = findEnvelopeTensor(outputs, name);
  if (tensor == nullptr) {
    throw std::runtime_error("Envelope is missing output " + name);
  }
  if (tensor->datatype != datatype) {
    throw std::runtime_error("Envelope output " + name + " is " +
                             tensor->datatype + ", expected " + datatype);
  }
  return tensor;
}

template <typename T>
T envelopeValueAt(const kserve::InferOutput &tensor, size_t index) {
  const size_t offset = index * sizeof(T);
  if (offset + sizeof(T) > tensor.data.size()) {
    throw std::runtime_error("Envelope output " + tensor.name +
                             " is truncated");
  }
  T value{};
  std::memcpy(&value, tensor.data.data() + offset, sizeof(T));
  return value;
}

// True when the model returns a decoded envelope rather than the inner model's
// raw tensors, i.e. the server postprocessed.
inline bool isDecodedEnvelope(const kserve::ModelMetadata &metadata) {
  for (const auto &output : metadata.outputs) {
    if (output.name == "NUM_DETECTIONS") {
      return true;
    }
  }
  return false;
}

inline EnvelopeVariant
envelopeVariantOf(const kserve::ModelMetadata &metadata) {
  for (const auto &output : metadata.outputs) {
    if (output.name == "POLYGON_POINTS") {
      return EnvelopeVariant::Polygon;
    }
    if (output.name == "MASK_DATA") {
      return EnvelopeVariant::Mask;
    }
  }
  return EnvelopeVariant::Detection;
}

// Validates the envelope the model declares, before any inference runs.
inline void validateEnvelopeModel(const kserve::ModelMetadata &metadata,
                                  EnvelopeVariant variant) {
  if (metadata.inputs.size() != 1 || metadata.inputs[0].name != "IMAGE" ||
      metadata.inputs[0].datatype != "UINT8") {
    throw std::runtime_error(
        "Envelope model must expose one UINT8 IMAGE input");
  }

  struct Expected {
    const char *name;
    const char *datatype;
  };
  std::vector<Expected> expected = {{"NUM_DETECTIONS", "INT32"},
                                    {"BOXES", "INT32"},
                                    {"SCORES", "FP32"},
                                    {"CLASSES", "INT32"}};
  if (variant == EnvelopeVariant::Mask) {
    expected.push_back({"MASK_OFFSETS", "INT64"});
    expected.push_back({"MASK_DATA", "UINT8"});
  } else if (variant == EnvelopeVariant::Polygon) {
    expected.push_back({"INSTANCE_RING_OFFSETS", "INT64"});
    expected.push_back({"RING_POINT_OFFSETS", "INT64"});
    expected.push_back({"POLYGON_POINTS", "INT32"});
  }

  for (const auto &item : expected) {
    bool found = false;
    for (const auto &output : metadata.outputs) {
      if (output.name != item.name) {
        continue;
      }
      found = true;
      if (output.datatype != item.datatype) {
        throw std::runtime_error(std::string("Envelope output mismatch: ") +
                                 item.name);
      }
      break;
    }
    if (!found) {
      throw std::runtime_error(std::string("Missing envelope output: ") +
                               item.name);
    }
  }
}

inline int readDetectionCount(const std::vector<kserve::InferOutput> &outputs) {
  const auto *count = requireEnvelopeTensor(outputs, "NUM_DETECTIONS", "INT32");
  const auto value = envelopeValueAt<int32_t>(*count, 0);
  if (value < 0 || value > kEnvelopeMaxDetections) {
    throw std::runtime_error("NUM_DETECTIONS is out of range: " +
                             std::to_string(value));
  }
  return value;
}

// Decodes the detection envelope. Boxes are already in source-image pixels, so
// no inverse letterbox is needed here -- the server did it.
inline std::vector<neuriplo_tasks::Result>
decodeDetectionEnvelope(const std::vector<kserve::InferOutput> &outputs) {
  const int count = readDetectionCount(outputs);
  const auto *boxes = requireEnvelopeTensor(outputs, "BOXES", "INT32");
  const auto *scores = requireEnvelopeTensor(outputs, "SCORES", "FP32");
  const auto *classes = requireEnvelopeTensor(outputs, "CLASSES", "INT32");

  std::vector<neuriplo_tasks::Result> results;
  results.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const auto index = static_cast<size_t>(i);
    neuriplo_tasks::Detection detection;
    detection.bbox.x = envelopeValueAt<int32_t>(*boxes, index * 4);
    detection.bbox.y = envelopeValueAt<int32_t>(*boxes, index * 4 + 1);
    detection.bbox.width = envelopeValueAt<int32_t>(*boxes, index * 4 + 2);
    detection.bbox.height = envelopeValueAt<int32_t>(*boxes, index * 4 + 3);
    detection.class_confidence = envelopeValueAt<float>(*scores, index);
    detection.class_id =
        static_cast<float>(envelopeValueAt<int32_t>(*classes, index));
    results.emplace_back(std::move(detection));
  }
  return results;
}

// Decodes the packed-mask segmentation envelope. MASK_OFFSETS is a prefix-sum
// array of 101 entries whatever the detection count; a shorter one means the
// server truncated it, which is the empty-frame defect this check exists for.
inline std::vector<neuriplo_tasks::Result>
decodeMaskEnvelope(const std::vector<kserve::InferOutput> &outputs,
                   int frame_width = 0, int frame_height = 0) {
  const int count = readDetectionCount(outputs);
  const auto *boxes = requireEnvelopeTensor(outputs, "BOXES", "INT32");
  const auto *scores = requireEnvelopeTensor(outputs, "SCORES", "FP32");
  const auto *classes = requireEnvelopeTensor(outputs, "CLASSES", "INT32");
  const auto *offsets = requireEnvelopeTensor(outputs, "MASK_OFFSETS", "INT64");
  const auto *mask_data = requireEnvelopeTensor(outputs, "MASK_DATA", "UINT8");

  if (offsets->data.size() <
      static_cast<size_t>(kEnvelopeMaxDetections + 1) * sizeof(int64_t)) {
    throw std::runtime_error("MASK_OFFSETS is truncated");
  }
  if (envelopeValueAt<int64_t>(*offsets, 0) != 0) {
    throw std::runtime_error("MASK_OFFSETS must begin at zero");
  }

  std::vector<neuriplo_tasks::Result> results;
  results.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const auto index = static_cast<size_t>(i);
    neuriplo_tasks::InstanceSegmentation segmentation;
    segmentation.bbox.x = envelopeValueAt<int32_t>(*boxes, index * 4);
    segmentation.bbox.y = envelopeValueAt<int32_t>(*boxes, index * 4 + 1);
    segmentation.bbox.width = envelopeValueAt<int32_t>(*boxes, index * 4 + 2);
    segmentation.bbox.height = envelopeValueAt<int32_t>(*boxes, index * 4 + 3);
    segmentation.class_confidence = envelopeValueAt<float>(*scores, index);
    segmentation.class_id =
        static_cast<float>(envelopeValueAt<int32_t>(*classes, index));

    const auto begin = envelopeValueAt<int64_t>(*offsets, index);
    const auto end = envelopeValueAt<int64_t>(*offsets, index + 1);
    if (end < begin || static_cast<size_t>(end) > mask_data->data.size()) {
      throw std::runtime_error("MASK_OFFSETS run past MASK_DATA");
    }
    segmentation.mask_data.assign(mask_data->data.begin() + begin,
                                  mask_data->data.begin() + end);
    segmentation.mask_width = static_cast<int>(segmentation.bbox.width);
    segmentation.mask_height = static_cast<int>(segmentation.bbox.height);

    // Rebuild the renderable mask image. The envelope carries no per-mask
    // dimensions because the contract fixes a mask to its detection's box, so
    // the box is what makes the byte run readable. Renderers draw the image,
    // not the raw bytes, so skipping this leaves masks invisible.
    //
    // The image is built at frame size with the mask placed at the box origin,
    // matching what the local YOLO postprocessor produces. A box-sized image
    // would be stretched across the whole frame by the renderer's resize.
    const auto expected = static_cast<size_t>(segmentation.mask_width) *
                          static_cast<size_t>(segmentation.mask_height);
    if (segmentation.mask_width > 0 && segmentation.mask_height > 0 &&
        segmentation.mask_data.size() == expected && frame_width > 0 &&
        frame_height > 0) {
      auto mask = neuriplo_tasks::vision::Image::zeros(
          frame_width, frame_height, 1,
          neuriplo_tasks::vision::PixelType::UInt8);
      auto *pixels = mask.data<uint8_t>();
      const auto origin_x = static_cast<int>(segmentation.bbox.x);
      const auto origin_y = static_cast<int>(segmentation.bbox.y);
      for (int row = 0; row < segmentation.mask_height; ++row) {
        const int target_row = origin_y + row;
        if (target_row < 0 || target_row >= frame_height) {
          continue;
        }
        for (int column = 0; column < segmentation.mask_width; ++column) {
          const int target_column = origin_x + column;
          if (target_column < 0 || target_column >= frame_width) {
            continue;
          }
          pixels[static_cast<size_t>(target_row) *
                     static_cast<size_t>(frame_width) +
                 static_cast<size_t>(target_column)] =
              segmentation
                  .mask_data[static_cast<size_t>(row) *
                                 static_cast<size_t>(segmentation.mask_width) +
                             static_cast<size_t>(column)];
        }
      }
      segmentation.mask = neuriplo_tasks::fromImage(std::move(mask));
      segmentation.mask_width = frame_width;
      segmentation.mask_height = frame_height;
    }
    results.emplace_back(std::move(segmentation));
  }
  return results;
}

// Even-odd ray cast, used to tell a hole from a separate exterior: the envelope
// carries rings as a flat list with no nesting flag.
inline bool
pointInsideRing(const std::vector<neuriplo_tasks::vision::Point2f> &ring,
                const neuriplo_tasks::vision::Point2f &point) {
  bool inside = false;
  for (size_t current = 0, previous = ring.size() - 1; current < ring.size();
       previous = current++) {
    const auto &a = ring[current];
    const auto &b = ring[previous];
    const bool crosses = (a.y > point.y) != (b.y > point.y);
    if (crosses &&
        point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x) {
      inside = !inside;
    }
  }
  return inside;
}

// Decodes the polygon segmentation envelope. Two levels of prefix offsets:
// detection to its rings, ring to its points. A ring contained inside an
// earlier ring of the same detection is a hole in it.
inline std::vector<neuriplo_tasks::Result>
decodePolygonEnvelope(const std::vector<kserve::InferOutput> &outputs) {
  const int count = readDetectionCount(outputs);
  const auto *boxes = requireEnvelopeTensor(outputs, "BOXES", "INT32");
  const auto *scores = requireEnvelopeTensor(outputs, "SCORES", "FP32");
  const auto *classes = requireEnvelopeTensor(outputs, "CLASSES", "INT32");
  const auto *instance_offsets =
      requireEnvelopeTensor(outputs, "INSTANCE_RING_OFFSETS", "INT64");
  const auto *ring_offsets =
      requireEnvelopeTensor(outputs, "RING_POINT_OFFSETS", "INT64");
  const auto *points =
      requireEnvelopeTensor(outputs, "POLYGON_POINTS", "INT32");

  if (instance_offsets->data.size() <
      static_cast<size_t>(kEnvelopeMaxDetections + 1) * sizeof(int64_t)) {
    throw std::runtime_error("INSTANCE_RING_OFFSETS is truncated");
  }
  if (envelopeValueAt<int64_t>(*instance_offsets, 0) != 0) {
    throw std::runtime_error("INSTANCE_RING_OFFSETS must begin at zero");
  }

  std::vector<neuriplo_tasks::Result> results;
  results.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const auto index = static_cast<size_t>(i);
    neuriplo_tasks::InstanceSegmentation segmentation;
    segmentation.bbox.x = envelopeValueAt<int32_t>(*boxes, index * 4);
    segmentation.bbox.y = envelopeValueAt<int32_t>(*boxes, index * 4 + 1);
    segmentation.bbox.width = envelopeValueAt<int32_t>(*boxes, index * 4 + 2);
    segmentation.bbox.height = envelopeValueAt<int32_t>(*boxes, index * 4 + 3);
    segmentation.class_confidence = envelopeValueAt<float>(*scores, index);
    segmentation.class_id =
        static_cast<float>(envelopeValueAt<int32_t>(*classes, index));

    const auto first_ring = envelopeValueAt<int64_t>(*instance_offsets, index);
    const auto last_ring =
        envelopeValueAt<int64_t>(*instance_offsets, index + 1);
    if (last_ring < first_ring) {
      throw std::runtime_error("INSTANCE_RING_OFFSETS are not monotonic");
    }

    for (int64_t ring = first_ring; ring < last_ring; ++ring) {
      const auto begin =
          envelopeValueAt<int64_t>(*ring_offsets, static_cast<size_t>(ring));
      const auto end = envelopeValueAt<int64_t>(*ring_offsets,
                                                static_cast<size_t>(ring) + 1);
      if (end < begin) {
        throw std::runtime_error("RING_POINT_OFFSETS are not monotonic");
      }

      std::vector<neuriplo_tasks::vision::Point2f> ring_points;
      ring_points.reserve(static_cast<size_t>(end - begin));
      for (int64_t point = begin; point < end; ++point) {
        const auto x =
            envelopeValueAt<int32_t>(*points, static_cast<size_t>(point) * 2);
        const auto y = envelopeValueAt<int32_t>(
            *points, static_cast<size_t>(point) * 2 + 1);
        ring_points.push_back({static_cast<float>(x), static_cast<float>(y)});
      }
      if (ring_points.size() < 3) {
        throw std::runtime_error(
            "Envelope polygon ring has fewer than three points");
      }
      // The first ring of a detection is its exterior; later rings that fall
      // inside an existing exterior are its holes. The owning polygon is
      // resolved before the ring is moved, so the move happens exactly once
      // down either path.
      neuriplo_tasks::SegmentationPolygon *container = nullptr;
      for (auto &polygon : segmentation.polygons) {
        if (pointInsideRing(polygon.exterior, ring_points.front())) {
          container = &polygon;
          break;
        }
      }
      if (container != nullptr) {
        container->holes.push_back(std::move(ring_points));
      } else {
        neuriplo_tasks::SegmentationPolygon polygon;
        polygon.exterior = std::move(ring_points);
        segmentation.polygons.push_back(std::move(polygon));
      }
    }

    results.emplace_back(std::move(segmentation));
  }
  return results;
}

inline std::vector<neuriplo_tasks::Result>
decodeEnvelope(const std::vector<kserve::InferOutput> &outputs,
               EnvelopeVariant variant, int frame_width = 0,
               int frame_height = 0) {
  switch (variant) {
  case EnvelopeVariant::Mask:
    return decodeMaskEnvelope(outputs, frame_width, frame_height);
  case EnvelopeVariant::Polygon:
    return decodePolygonEnvelope(outputs);
  case EnvelopeVariant::Detection:
    break;
  }
  return decodeDetectionEnvelope(outputs);
}

} // namespace neuriplo_infer
