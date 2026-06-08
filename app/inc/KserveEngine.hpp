#pragma once

// Adapter that exposes a pure KServe protocol client (kserve::IClient) as a
// neuriplo InferenceInterface, so a remote KServe model is a drop-in for a local
// engine inside InferencePipeline.
//
// This is the ONLY KServe file that depends on the neuriplo contract
// (InferenceInterface / TensorElement / InferenceMetadata). The protocol clients
// themselves are neuriplo-free; this layer owns metadata caching and the
// conversion between raw protocol bytes and the application's typed tensors.

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

private:
  void ensureMetadata();

  std::unique_ptr<kserve::IClient> client_;
  bool metadata_loaded_{false};
  kserve::ModelMetadata raw_metadata_;
  InferenceMetadata cached_metadata_;
};
