#pragma once

// Adapter that exposes a pure KServe protocol client (kserve::IClient) as a
// neuriplo InferenceInterface, so a remote KServe model is a drop-in for a
// local engine inside InferencePipeline.
//
// This is the ONLY KServe file that depends on the neuriplo contract
// (InferenceInterface / TensorElement / InferenceMetadata). The protocol
// clients themselves are neuriplo-free; this layer owns metadata caching and
// the conversion between raw protocol bytes and the application's typed
// tensors.

#include "InferenceInterface.hpp"
#include "InferenceMetadata.hpp"
#include "KserveTypes.hpp"

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

class KserveEngine : public InferenceInterface {
public:
  explicit KserveEngine(std::unique_ptr<kserve::IClient> client);

  std::tuple<std::vector<std::vector<TensorElement>>,
             std::vector<std::vector<int64_t>>>
  get_infer_results(
      const std::vector<std::vector<uint8_t>> &input_tensors) override;

  InferenceMetadata get_inference_metadata() override;

  bool is_gpu_available() const noexcept override;

  // Observability: per-request inference latency, measured around the remote
  // round-trip (kserve::IClient::infer()). The task pipeline times the whole
  // request; these expose the network/inference component so a caller can
  // attribute remote cost. Each infer() also emits a debug log line
  // (glog VLOG(1)). All values are in milliseconds.
  //   lastInferenceLatencyMs    — most recent infer() round-trip
  //   averageInferenceLatencyMs — mean over all infer() calls (0 if none yet)
  //   inferenceCount            — number of infer() calls completed
  double lastInferenceLatencyMs() const noexcept;
  double averageInferenceLatencyMs() const noexcept;
  uint64_t inferenceCount() const noexcept;

private:
  void ensureMetadata();

  std::unique_ptr<kserve::IClient> client_;
  bool metadata_loaded_{false};
  kserve::ModelMetadata raw_metadata_;
  InferenceMetadata cached_metadata_;

  double last_latency_ms_{0.0};
  double total_latency_ms_{0.0};
  uint64_t infer_count_{0};
};
