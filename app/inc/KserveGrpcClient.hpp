#pragma once

#include "InferenceInterface.hpp"
#include "InferenceMetadata.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace grpc_client {

class KserveGrpcClient : public InferenceInterface {
public:
  KserveGrpcClient(const std::string &endpoint, const std::string &model_name,
                   const std::string &model_version = "1",
                   int timeout_ms = 30000);
  ~KserveGrpcClient() override;

  std::tuple<std::vector<std::vector<TensorElement>>,
             std::vector<std::vector<int64_t>>>
  get_infer_results(
      const std::vector<std::vector<uint8_t>> &input_tensors) override;

  InferenceMetadata get_inference_metadata() override;

  bool is_gpu_available() const noexcept override;

private:
  bool fetchMetadata();

  std::string endpoint_;
  std::string model_name_;
  std::string model_version_;
  int timeout_ms_;
  bool metadata_loaded_{false};
  InferenceMetadata cached_metadata_;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace grpc_client
