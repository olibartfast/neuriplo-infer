#pragma once

// Neutral data types and the pure protocol-client contract for the KServe V2 /
// Open Inference Protocol. Nothing here depends on the neuriplo backend types
// (no TensorElement, no InferenceInterface), so a protocol client built on this
// header is a standalone peer of e.g. Triton's client library: the only shared
// contract with the server is the wire protocol, not a C++ class.
//
// Tensor payloads are carried as raw little-endian bytes (like Triton's
// InferResult raw data); converting bytes to the application's typed tensors is
// the job of the adapter layer (KserveEngine), not the client.

#include <cstdint>
#include <string>
#include <vector>

namespace kserve {

// Name + datatype tag + shape of one model input or output.
struct TensorSpec {
  std::string name;
  std::string datatype; // KServe tag, e.g. "FP32", "INT64"
  std::vector<int64_t> shape;
};

// Model input/output description as reported by the server.
struct ModelMetadata {
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
};

// One input tensor for an inference request. `data` points at preprocessed raw
// little-endian bytes owned by the caller for the duration of the infer() call.
struct InferInput {
  std::string name;
  std::string datatype;
  std::vector<int64_t> shape;
  const std::vector<uint8_t> *data = nullptr;
};

// One output tensor returned by the server, as raw little-endian bytes.
struct InferOutput {
  std::string name;
  std::string datatype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
};

// Pure protocol client: speaks KServe V2 over HTTP or gRPC. Implementations
// depend only on the wire protocol and standard transport libraries, never on
// neuriplo. State such as metadata caching belongs to the caller.
class IClient {
public:
  virtual ~IClient() = default;

  // Fetches the model's input/output description from the server.
  virtual ModelMetadata modelMetadata() = 0;

  // Runs inference and returns every output the server produced.
  virtual std::vector<InferOutput>
  infer(const std::vector<InferInput> &inputs) = 0;
};

} // namespace kserve
