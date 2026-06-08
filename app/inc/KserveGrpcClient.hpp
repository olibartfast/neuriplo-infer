#pragma once

// Pure gRPC protocol client for KServe V2 / Open Inference Protocol. Depends
// only on KserveTypes (and the generated protobuf/gRPC stubs in the .cpp). No
// neuriplo dependency — see KserveTypes.hpp for the rationale.

#include "KserveTypes.hpp"

#include <memory>
#include <string>

namespace kserve {

class GrpcClient : public IClient {
public:
  GrpcClient(const std::string &endpoint, std::string model_name,
             std::string model_version = "1", int timeout_ms = 30000);
  ~GrpcClient() override;

  ModelMetadata modelMetadata() override;
  std::vector<InferOutput>
  infer(const std::vector<InferInput> &inputs) override;

  bool serverLive() override;
  bool serverReady() override;
  bool modelReady() override;

private:
  std::string model_name_;
  std::string model_version_;
  int timeout_ms_;
  // Optional bearer token, sourced from the KSERVE_BEARER_TOKEN env var.
  std::string auth_token_;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace kserve
