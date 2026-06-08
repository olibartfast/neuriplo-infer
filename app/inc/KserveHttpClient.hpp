#pragma once

// Pure HTTP protocol client for KServe V2 / Open Inference Protocol. Depends
// only on KserveTypes / KserveProtocol (and nlohmann/json in the .cpp). No
// neuriplo dependency — see KserveTypes.hpp for the rationale.

#include "KserveRetry.hpp"
#include "KserveTypes.hpp"

#include <memory>
#include <string>
#include <vector>

namespace kserve {

// Persistent keep-alive socket, defined in the .cpp (pImpl: keeps the socket
// headers out of this header).
struct HttpConnection;

class HttpClient : public IClient {
public:
  HttpClient(std::string endpoint, std::string model_name,
             std::string model_version = "1", int timeout_ms = 30000);
  ~HttpClient() override;

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
  // Retry/backoff policy for transient failures, sourced from the environment.
  RetryPolicy retry_policy_;
  // Output names from the last modelMetadata() call, used to request binary
  // outputs in infer() when the binary extension is enabled.
  std::vector<std::string> output_names_;
  // Persistent keep-alive socket. Reused across requests; dropped and reopened
  // on any I/O error.
  std::shared_ptr<HttpConnection> conn_;
};

} // namespace kserve
