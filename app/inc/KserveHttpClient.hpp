#pragma once

// Pure HTTP protocol client for KServe V2 / Open Inference Protocol. Depends
// only on KserveTypes / KserveProtocol (and nlohmann/json in the .cpp). No
// neuriplo dependency — see KserveTypes.hpp for the rationale.

#include "KserveTypes.hpp"

#include <string>

namespace kserve {

class HttpClient : public IClient {
public:
  HttpClient(std::string endpoint, std::string model_name,
             std::string model_version = "1", int timeout_ms = 30000);

  ModelMetadata modelMetadata() override;
  std::vector<InferOutput>
  infer(const std::vector<InferInput> &inputs) override;

  bool serverLive() override;
  bool serverReady() override;
  bool modelReady() override;

private:
  // GETs `path_prefix + path` and reports whether the server answered HTTP 200.
  bool probe(const std::string &path);

  std::string endpoint_;
  std::string model_name_;
  std::string model_version_;
  int timeout_ms_;
  // Optional bearer token, sourced from the KSERVE_BEARER_TOKEN env var.
  std::string auth_token_;
};

} // namespace kserve
