#include "ModelInputTypes.hpp"

#include <algorithm>
#include <stdexcept>

namespace neuriplo_infer {

std::vector<int64_t> taskInputShape(std::vector<int64_t> shape,
                                    size_t batch_size,
                                    bool video_classification) {
  if (shape.size() == 3 || (video_classification && shape.size() == 4)) {
    shape.insert(shape.begin(), static_cast<int64_t>(batch_size));
  }
  return shape;
}

std::string datatypeTag(TensorDataType datatype) {
  switch (datatype) {
  case TensorDataType::Float32:
    return "FP32";
  case TensorDataType::Int32:
    return "INT32";
  case TensorDataType::Int64:
    return "INT64";
  case TensorDataType::UInt8:
    return "UINT8";
  case TensorDataType::Int8:
    return "INT8";
  case TensorDataType::Bool:
    return "BOOL";
  }
  return "UNKNOWN";
}

void applyInputDatatypes(neuriplo_tasks::ModelInfo &model_info,
                         const std::vector<std::string> &datatypes,
                         bool server_preprocesses) {
  if (server_preprocesses) {
    return;
  }

  if (datatypes.size() != model_info.input_shapes.size()) {
    throw std::runtime_error(
        "internal error: " + std::to_string(datatypes.size()) +
        " input datatypes for " +
        std::to_string(model_info.input_shapes.size()) +
        " model inputs; refusing to guess the type of an image input");
  }

  // Preserve every tensor's actual runtime type, including auxiliary inputs
  // whose scalar types cannot be represented by the image PixelType enum.
  model_info.input_datatypes = datatypes;

  const size_t count =
      std::min({model_info.input_shapes.size(), model_info.input_types.size(),
                datatypes.size()});
  for (size_t i = 0; i < count; ++i) {
    if (!neuriplo_tasks::isImageInputShape(model_info.input_shapes[i])) {
      continue;
    }
    const auto &datatype = datatypes[i];
    if (datatype == "FP32") {
      model_info.input_types[i] = neuriplo_tasks::PixelType::Float32;
    } else if (datatype == "UINT8") {
      model_info.input_types[i] = neuriplo_tasks::PixelType::UInt8;
    } else {
      std::string message = "model input '";
      message += i < model_info.input_names.size() ? model_info.input_names[i]
                                                   : "#" + std::to_string(i);
      message += "' is ";
      message += datatype;
      message +=
          ", but image preprocessing produces FP32 or UINT8 tensors; serve the "
          "model with an FP32 or UINT8 image input, or let a server-side "
          "ensemble preprocess it (--input_mode=encoded-image)";
      throw std::runtime_error(message);
    }
  }
}

} // namespace neuriplo_infer
