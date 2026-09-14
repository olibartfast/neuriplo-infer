#include "KserveEngine.hpp"

#include "KserveProtocol.hpp"

#include <glog/logging.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

size_t batchSize(const std::vector<int64_t> &shape) {
  if (!shape.empty() && shape[0] > 0) {
    return static_cast<size_t>(shape[0]);
  }
  return 1;
}

// The metadata enum has no FP16 / FP64 / INT16 / UINT16+ members; those keep
// the Float32 default here, and a caller that must tell them apart reads
// rawMetadata().
TensorDataType toTensorDataType(const std::string &datatype) {
  if (datatype == "INT32") {
    return TensorDataType::Int32;
  }
  if (datatype == "INT64") {
    return TensorDataType::Int64;
  }
  if (datatype == "UINT8") {
    return TensorDataType::UInt8;
  }
  if (datatype == "INT8") {
    return TensorDataType::Int8;
  }
  if (datatype == "BOOL") {
    return TensorDataType::Bool;
  }
  return TensorDataType::Float32;
}

// Refuses a tensor whose bytes cannot be whole `spec.datatype` elements of
// `spec.shape`, so bytes preprocessed as one type are never sent labelled as
// another. A negative (dynamic) dimension accepts any extent.
void validateInputBytes(const kserve::TensorSpec &spec, size_t byte_count) {
  const size_t width = kserve::datatypeByteWidth(spec.datatype);
  if (width == 0) {
    return; // BYTES and unknown tags have no fixed element width.
  }

  size_t fixed_bytes = width;
  bool dynamic = false;
  for (const auto dim : spec.shape) {
    if (dim < 0) {
      dynamic = true;
      continue;
    }
    const size_t extent = static_cast<size_t>(dim);
    if (extent != 0 &&
        fixed_bytes > std::numeric_limits<size_t>::max() / extent) {
      throw std::runtime_error("KServe input '" + spec.name +
                               "' has a shape whose element count overflows "
                               "the expected byte count");
    }
    fixed_bytes *= extent;
  }
  const bool fits = dynamic
                        ? (fixed_bytes == 0 || byte_count % fixed_bytes == 0)
                        : byte_count == fixed_bytes;
  if (fits) {
    return;
  }

  std::string dims;
  for (const auto dim : spec.shape) {
    dims += (dims.empty() ? "" : ",") + std::to_string(dim);
  }
  throw std::runtime_error(
      "KServe input '" + spec.name + "' is " + spec.datatype + " [" + dims +
      "] and expects " + (dynamic ? "a multiple of " : "") +
      std::to_string(fixed_bytes) + " bytes, but the request contains " +
      std::to_string(byte_count) +
      "; the model's input datatype does not match the supplied bytes");
}

// Reads raw little-endian bytes as `T` and appends each value, projected to a
// TensorElement alternative, to `out`.
template <typename T, typename Proj>
void readInto(const std::vector<uint8_t> &bytes,
              std::vector<TensorElement> &out, Proj proj) {
  if (bytes.size() % sizeof(T) != 0) {
    throw std::runtime_error(
        "KServe output byte count is not a multiple of the "
        "datatype width");
  }
  const size_t count = bytes.size() / sizeof(T);
  for (size_t i = 0; i < count; ++i) {
    T value;
    std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));
    out.push_back(proj(value));
  }
}

// Decodes raw protocol bytes into the typed TensorElement variant. The widening
// choices mirror the variant's alternatives (float / int32 / int64 / uint8).
std::vector<TensorElement> bytesToElements(const std::vector<uint8_t> &bytes,
                                           const std::string &datatype) {
  std::vector<TensorElement> out;
  if (datatype == "FP32") {
    readInto<float>(bytes, out, [](float v) { return v; });
  } else if (datatype == "FP64") {
    readInto<double>(bytes, out,
                     [](double v) { return static_cast<float>(v); });
  } else if (datatype == "FP16") {
    readInto<uint16_t>(bytes, out,
                       [](uint16_t v) { return kserve::halfToFloat(v); });
  } else if (datatype == "INT8") {
    readInto<int8_t>(bytes, out,
                     [](int8_t v) { return static_cast<int32_t>(v); });
  } else if (datatype == "INT16") {
    readInto<int16_t>(bytes, out,
                      [](int16_t v) { return static_cast<int32_t>(v); });
  } else if (datatype == "INT32") {
    readInto<int32_t>(bytes, out, [](int32_t v) { return v; });
  } else if (datatype == "INT64") {
    readInto<int64_t>(bytes, out, [](int64_t v) { return v; });
  } else if (datatype == "UINT8" || datatype == "BOOL") {
    readInto<uint8_t>(bytes, out, [](uint8_t v) { return v; });
  } else if (datatype == "UINT16") {
    readInto<uint16_t>(bytes, out,
                       [](uint16_t v) { return static_cast<int64_t>(v); });
  } else if (datatype == "UINT32") {
    readInto<uint32_t>(bytes, out,
                       [](uint32_t v) { return static_cast<int64_t>(v); });
  } else if (datatype == "UINT64") {
    readInto<uint64_t>(bytes, out,
                       [](uint64_t v) { return static_cast<int64_t>(v); });
  } else {
    throw std::runtime_error("unsupported KServe output datatype: " + datatype);
  }
  return out;
}

// Metadata shapes may carry a negative (dynamic) dimension; the wire needs a
// concrete one. Encoded-image inputs are the case that forces this: the byte
// length is different for every request, so it can only come from the payload.
std::vector<int64_t> concreteInputShape(const kserve::TensorSpec &spec,
                                        size_t byte_count) {
  std::vector<int64_t> shape = spec.shape;
  const auto width = kserve::datatypeByteWidth(spec.datatype);
  if (width == 0) {
    return shape;
  }

  int64_t fixed = 1;
  size_t dynamic_axes = 0;
  size_t dynamic_index = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      ++dynamic_axes;
      dynamic_index = i;
    } else {
      if (shape[i] != 0 &&
          fixed > std::numeric_limits<int64_t>::max() / shape[i]) {
        return shape; // Overflow: leave the dynamic axis uninferred.
      }
      fixed *= shape[i];
    }
  }
  // Only a single dynamic axis can be inferred from the byte count alone.
  if (dynamic_axes != 1 || fixed <= 0) {
    return shape;
  }

  const auto elements = static_cast<int64_t>(byte_count / width);
  shape[dynamic_index] = elements / fixed;
  return shape;
}

// --input_sizes gives CHW extents without the batch axis. They fill the dynamic
// trailing dimensions of the matching metadata shape, which the payload size
// alone cannot resolve when more than one is dynamic.
void applyInputSizes(std::vector<int64_t> &shape,
                     const std::vector<std::vector<int64_t>> &input_sizes,
                     size_t index) {
  if (index >= input_sizes.size()) {
    return;
  }
  const auto &sizes = input_sizes[index];
  if (sizes.empty() || sizes.size() > shape.size()) {
    return;
  }
  const size_t offset = shape.size() - sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (shape[offset + i] < 0) {
      shape[offset + i] = sizes[i];
    }
  }
}

// Postprocessors index output tensors by their reported shape, so a payload
// shorter than that shape would be read past its end. Refused here instead,
// naming the output. Variable-width tags (BYTES) carry no element size.
void validateOutputBytes(const kserve::InferOutput &output) {
  const size_t width = kserve::datatypeByteWidth(output.datatype);
  if (width == 0) {
    return;
  }
  size_t elements = 1;
  std::string dims;
  for (const auto dim : output.shape) {
    dims += (dims.empty() ? "" : ",") + std::to_string(dim);
  }
  for (const auto dim : output.shape) {
    if (dim < 0) {
      throw std::runtime_error("KServe output '" + output.name + "' [" + dims +
                               "] has a negative dimension");
    }
    const auto extent = static_cast<size_t>(dim);
    if (extent != 0 && elements > std::numeric_limits<size_t>::max() / extent) {
      throw std::runtime_error("KServe output '" + output.name + "' [" + dims +
                               "] has an element count that overflows");
    }
    elements *= extent;
  }
  if (elements > std::numeric_limits<size_t>::max() / width ||
      output.data.size() != elements * width) {
    throw std::runtime_error("KServe output '" + output.name + "' is " +
                             output.datatype + " [" + dims + "] and needs " +
                             std::to_string(elements * width) +
                             " bytes, but the response carries " +
                             std::to_string(output.data.size()));
  }
}

template <typename Metadata>
std::string metadataPlatform(const Metadata &metadata) {
  if constexpr (requires { metadata.platform; }) {
    return metadata.platform;
  } else {
    return {};
  }
}

} // namespace

KserveEngine::KserveEngine(std::unique_ptr<kserve::IClient> client)
    : KserveEngine(std::move(client), {}) {}

KserveEngine::KserveEngine(std::unique_ptr<kserve::IClient> client,
                           std::vector<std::vector<int64_t>> input_sizes)
    : InferenceInterface("", false, 1), client_(std::move(client)),
      input_sizes_(std::move(input_sizes)) {
  if (!client_) {
    throw std::runtime_error("KserveEngine requires a non-null client");
  }
}

void KserveEngine::ensureMetadata() {
  if (metadata_loaded_) {
    return;
  }
  // Fail fast with a clear message when the server is reachable but the model
  // is not loaded/ready. A transport failure (unreachable endpoint) propagates
  // from the probe as a connection error, distinct from "up but not ready".
  if (!client_->modelReady()) {
    throw std::runtime_error(
        "KServe server is reachable but the model is not ready "
        "(not loaded, or still initialising)");
  }
  raw_metadata_ = client_->modelMetadata();
  for (const auto &input : raw_metadata_.inputs) {
    cached_metadata_.addInput(input.name, input.shape, batchSize(input.shape),
                              toTensorDataType(input.datatype));
  }
  for (const auto &output : raw_metadata_.outputs) {
    cached_metadata_.addOutput(output.name, output.shape,
                               batchSize(output.shape),
                               toTensorDataType(output.datatype));
  }
  metadata_loaded_ = true;
}

InferenceMetadata KserveEngine::get_inference_metadata() {
  ensureMetadata();
  return cached_metadata_;
}

std::tuple<std::vector<std::vector<TensorElement>>,
           std::vector<std::vector<int64_t>>>
KserveEngine::get_infer_results(
    const std::vector<std::vector<uint8_t>> &input_tensors) {
  ensureMetadata();

  if (input_tensors.size() != raw_metadata_.inputs.size()) {
    throw std::runtime_error("KServe input tensor count mismatch");
  }

  std::vector<kserve::InferInput> inputs;
  std::vector<std::vector<int64_t>> shapes;
  inputs.reserve(input_tensors.size());
  shapes.reserve(input_tensors.size());
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    kserve::TensorSpec spec = raw_metadata_.inputs[i];
    applyInputSizes(spec.shape, input_sizes_, i);
    validateInputBytes(spec, input_tensors[i].size());
    shapes.push_back(concreteInputShape(spec, input_tensors[i].size()));
    // Only one dynamic axis can be inferred from the payload size, and none
    // for BYTES or other variable-width tags. A shape that still has one would
    // go on the wire with a -1 extent and fail on the server with an error that
    // no longer names the cause, so refuse it here for every datatype.
    for (const auto dim : shapes.back()) {
      if (dim < 0) {
        throw std::runtime_error(
            "KServe input '" + spec.name + "' (" + spec.datatype +
            ") has a dynamic dimension that cannot be inferred from the "
            "payload size (more than one dynamic axis, a variable-width "
            "datatype, or fixed dimensions that are zero or overflow); serve "
            "the model with concrete extents for that input");
      }
    }
    inputs.push_back(
        {spec.name, spec.datatype, shapes.back(), &input_tensors[i]});
  }

  const auto start = std::chrono::steady_clock::now();
  const auto results = client_->infer(inputs);
  const auto end = std::chrono::steady_clock::now();
  last_latency_ms_ =
      std::chrono::duration<double, std::milli>(end - start).count();
  total_latency_ms_ += last_latency_ms_;
  ++infer_count_;
  VLOG(1) << "KServe infer round-trip: " << last_latency_ms_ << " ms (request "
          << infer_count_ << ", avg " << averageInferenceLatencyMs() << " ms)";

  std::vector<std::vector<TensorElement>> output_data;
  std::vector<std::vector<int64_t>> output_shapes;
  output_data.reserve(results.size());
  output_shapes.reserve(results.size());
  for (const auto &output : results) {
    validateOutputBytes(output);
    output_shapes.push_back(output.shape);
    output_data.push_back(bytesToElements(output.data, output.datatype));
  }
  // Kept so an ensemble caller can decode the result envelope by tensor name;
  // the typed tuple above drops names, and the envelope is name-addressed.
  last_raw_outputs_ = results;

  return {std::move(output_data), std::move(output_shapes)};
}

bool KserveEngine::is_gpu_available() const noexcept { return false; }

double KserveEngine::lastInferenceLatencyMs() const noexcept {
  return last_latency_ms_;
}

double KserveEngine::averageInferenceLatencyMs() const noexcept {
  return infer_count_ == 0
             ? 0.0
             : total_latency_ms_ / static_cast<double>(infer_count_);
}

uint64_t KserveEngine::inferenceCount() const noexcept { return infer_count_; }

std::string KserveEngine::servingPlatform() const noexcept {
  return metadataPlatform(raw_metadata_);
}

const std::vector<kserve::InferOutput> &
KserveEngine::lastRawOutputs() const noexcept {
  return last_raw_outputs_;
}

const kserve::ModelMetadata &KserveEngine::rawMetadata() {
  ensureMetadata();
  return raw_metadata_;
}
