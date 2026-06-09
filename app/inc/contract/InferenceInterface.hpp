#pragma once

// App-local, dependency-free copy of the subset of neuriplo's
// InferenceInterface that neuriplo-infer relies on. Used when building WITHOUT
// local backends (KServe-only), so the app needs to fetch neither neuriplo nor
// any external contract library. Put on the include path only in that build
// mode; full builds use neuriplo's own header. Signatures must stay identical
// to neuriplo's so the same source compiles unchanged in both modes. See
// docs/KserveRoadmap.md.

#include "InferenceMetadata.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

class InferenceInterface {
public:
  InferenceInterface(const std::string &weights, bool use_gpu = false,
                     size_t batch_size = 1,
                     const std::vector<std::vector<int64_t>> &input_sizes =
                         std::vector<std::vector<int64_t>>())
      : model_path_(weights), gpu_available_(use_gpu), batch_size_(batch_size) {
    (void)input_sizes;
  }

  virtual ~InferenceInterface() = default;

  virtual std::tuple<std::vector<std::vector<TensorElement>>,
                     std::vector<std::vector<int64_t>>>
  get_infer_results(const std::vector<std::vector<uint8_t>> &input_tensors) = 0;

  virtual InferenceMetadata get_inference_metadata() {
    return inference_metadata_;
  }

  virtual bool is_gpu_available() const noexcept { return gpu_available_; }
  virtual size_t get_batch_size() const noexcept { return batch_size_; }
  virtual std::string get_model_path() const noexcept { return model_path_; }

protected:
  InferenceMetadata inference_metadata_;
  std::string model_path_;
  bool gpu_available_;
  size_t batch_size_;
};
