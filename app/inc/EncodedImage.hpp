#pragma once

// Encoded-image request path for server-side ensembles.
//
// Ported from tritonic v0.4.0 (include/tritonic/core/encoded_image.hpp) and
// retargeted from Triton's ModelInfo onto kserve::ModelMetadata, so the same
// application code works against our runtime and a Triton deployment. The wire
// surface is the platform ensemble contract.

#include "KserveTypes.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace neuriplo_infer {

struct EncodedImageRequest {
  std::vector<uint8_t> bytes;
  std::vector<int64_t> shape;
  int width{0};
  int height{0};
};

// Reads the source image's pixel dimensions from the JPEG header. The server
// does not return them and they are needed to map results back onto the
// original frame, so the client must read them itself.
inline std::pair<int, int> readJpegDimensions(const std::vector<uint8_t> &bytes,
                                              const std::string &path) {
  if (bytes.size() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8) {
    throw std::runtime_error("Encoded image is not a JPEG: " + path);
  }

  size_t position = 2;
  while (position + 3 < bytes.size()) {
    while (position < bytes.size() && bytes[position] != 0xff) {
      ++position;
    }
    while (position < bytes.size() && bytes[position] == 0xff) {
      ++position;
    }
    if (position >= bytes.size()) {
      break;
    }

    const uint8_t marker = bytes[position++];
    // Standalone markers carry no length field.
    if (marker == 0xd8 || marker == 0xd9 || marker == 0x01 ||
        (marker >= 0xd0 && marker <= 0xd7)) {
      continue;
    }
    if (position + 1 >= bytes.size()) {
      break;
    }

    const size_t length =
        (static_cast<size_t>(bytes[position]) << 8) | bytes[position + 1];
    if (length < 2 || position + length > bytes.size()) {
      break;
    }

    const bool startOfFrame = (marker >= 0xc0 && marker <= 0xc3) ||
                              (marker >= 0xc5 && marker <= 0xc7) ||
                              (marker >= 0xc9 && marker <= 0xcb) ||
                              (marker >= 0xcd && marker <= 0xcf);
    if (startOfFrame) {
      if (length < 7) {
        break;
      }
      const int height =
          (static_cast<int>(bytes[position + 3]) << 8) | bytes[position + 4];
      const int width =
          (static_cast<int>(bytes[position + 5]) << 8) | bytes[position + 6];
      if (width > 0 && height > 0) {
        return {width, height};
      }
      break;
    }
    position += length;
  }

  throw std::runtime_error("Could not read JPEG dimensions: " + path);
}

inline EncodedImageRequest buildEncodedImageRequest(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("Could not open encoded image: " + path);
  }

  std::vector<uint8_t> bytes(std::istreambuf_iterator<char>(stream), {});
  if (bytes.empty()) {
    throw std::runtime_error("Encoded image is empty: " + path);
  }

  const auto [width, height] = readJpegDimensions(bytes, path);
  const auto size = static_cast<int64_t>(bytes.size());
  return {std::move(bytes), {1, size}, width, height};
}

// Checks the ensemble against the inner task model.
//
// The ensemble's own metadata describes an encoded image, which tells the task
// layer nothing about tensor layout -- that comes from the inner model, fetched
// separately by name. This is where an ensemble wired to the wrong inner model
// is caught, and it is the reason --task_model exists.
inline void
validateEncodedImageModels(const kserve::ModelMetadata &ensemble,
                           const kserve::ModelMetadata &task_model) {
  if (ensemble.inputs.size() != 1 || ensemble.inputs[0].name != "IMAGE" ||
      ensemble.inputs[0].datatype != "UINT8") {
    throw std::runtime_error(
        "Encoded-image model must expose exactly one UINT8 input named IMAGE");
  }

  if (ensemble.outputs.size() < task_model.outputs.size()) {
    throw std::runtime_error(
        "Encoded-image model outputs do not match --task_model metadata");
  }
  // The ensemble's leading outputs must be the inner model's, unchanged; it may
  // append extra outputs after them.
  for (size_t i = 0; i < task_model.outputs.size(); ++i) {
    if (ensemble.outputs[i].name != task_model.outputs[i].name ||
        ensemble.outputs[i].datatype != task_model.outputs[i].datatype ||
        ensemble.outputs[i].shape != task_model.outputs[i].shape) {
      throw std::runtime_error(
          "Encoded-image model outputs do not match --task_model metadata");
    }
  }
}

} // namespace neuriplo_infer
