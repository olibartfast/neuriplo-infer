#pragma once

// Carries a model's advertised input datatypes into the task layer's ModelInfo.
// What preprocessing emits for each type is neuriplo-tasks' contract; this only
// translates backend and server datatypes into it, and refuses an image input
// no preprocessing path can fill before any inference request is built.

#include "InferenceMetadata.hpp"
#include "neuriplo/tasks/core/model_info.hpp"

#include <string>
#include <vector>

namespace neuriplo_infer {

// KServe datatype tag ("FP32", "UINT8", ...) for a backend metadata datatype.
std::string datatypeTag(TensorDataType datatype);

// Records each image input's datatype in model_info.input_types: FP32 and UINT8
// become their pixel types, and any other datatype on an image input throws
// std::runtime_error naming the input. `datatypes` holds one KServe tag per
// input, in model_info order. When the server preprocesses
// (--input_mode=encoded-image) nothing is checked or changed.
void applyInputDatatypes(neuriplo_tasks::ModelInfo &model_info,
                         const std::vector<std::string> &datatypes,
                         bool server_preprocesses);

} // namespace neuriplo_infer
