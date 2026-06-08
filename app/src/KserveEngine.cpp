#include "KserveEngine.hpp"

#include "KserveProtocol.hpp"

#include <glog/logging.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace {

size_t batchSize(const std::vector<int64_t> &shape) {
  if (!shape.empty() && shape[0] > 0) {
    return static_cast<size_t>(shape[0]);
  }
  return 1;
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

} // namespace

KserveEngine::KserveEngine(std::unique_ptr<kserve::IClient> client)
    : InferenceInterface("", false, 1), client_(std::move(client)) {
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
    cached_metadata_.addInput(input.name, input.shape, batchSize(input.shape));
  }
  for (const auto &output : raw_metadata_.outputs) {
    cached_metadata_.addOutput(output.name, output.shape,
                               batchSize(output.shape));
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
  inputs.reserve(input_tensors.size());
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    const auto &spec = raw_metadata_.inputs[i];
    inputs.push_back({spec.name, spec.datatype, spec.shape, &input_tensors[i]});
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
    output_shapes.push_back(output.shape);
    output_data.push_back(bytesToElements(output.data, output.datatype));
  }

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
