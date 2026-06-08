#pragma once

// Pure, dependency-light helpers for the KServe V2 / Open Inference Protocol.
// Nothing here depends on the neuriplo backend types, so it is unit-testable in
// isolation (only nlohmann/json + the standard library).

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kserve {

// Parsed connection target derived from a user-supplied endpoint URL.
struct Endpoint {
  std::string host;
  int port = 0;
  std::string path_prefix; // may be empty; never has a trailing '/'
  bool tls = false;        // true for https:// / grpcs://
};

// Parses "scheme://host:port/prefix" forms. Missing scheme is treated as
// plaintext. When no port is given, default_port is used.
Endpoint parseEndpoint(const std::string &url, int default_port);

// Result of an HTTP/1.1 response. body is fully de-chunked when the response
// used Transfer-Encoding: chunked.
struct HttpResponse {
  int status = 0;
  std::string body;
  // KServe Binary Tensor Data Extension: byte length of the JSON header that
  // prefixes the binary payload, taken from the response's
  // Inference-Header-Content-Length. -1 when the header is absent (plain JSON
  // response); 0 or more when the body is JSON-header + binary blob.
  long inference_header_length = -1;
};

// Parses a raw HTTP/1.1 response (headers + body). Honours Content-Length and
// Transfer-Encoding: chunked. Throws std::runtime_error on a malformed
// response.
HttpResponse parseHttpResponse(const std::string &raw);

// KServe V2 health/readiness endpoint paths, without any Endpoint::path_prefix
// (callers prepend it). modelReadyPath omits the /versions segment when version
// is empty, mirroring inferPath.
std::string serverLivePath();
std::string serverReadyPath();
std::string modelReadyPath(const std::string &model_name,
                           const std::string &model_version);

// Byte width of a KServe datatype tag (FP32, INT64, UINT8, ...). Returns 0 for
// variable-width / unknown datatypes (e.g. BYTES).
std::size_t datatypeByteWidth(const std::string &datatype);

// Converts an IEEE-754 half-precision bit pattern to float.
float halfToFloat(std::uint16_t h);

// Reinterprets preprocessed raw bytes as `datatype` and emits the JSON numeric
// array expected in a KServe V2 "data" field. Throws if the byte count is not a
// multiple of the datatype width. FP16 is widened to float for JSON.
nlohmann::json encodeTensorData(const std::vector<std::uint8_t> &bytes,
                                const std::string &datatype);

// Inverse of encodeTensorData: reads a KServe V2 "data" JSON numeric array and
// packs it into raw little-endian bytes for `datatype`. FP16 values are
// narrowed back to half. Throws on a non-array input or an unsupported
// datatype.
std::vector<std::uint8_t> decodeTensorData(const nlohmann::json &data,
                                           const std::string &datatype);

// --- KServe V2 Binary Tensor Data Extension ---------------------------------
//
// In the binary extension a request/response body is a JSON header immediately
// followed by the raw little-endian tensor bytes. The JSON header's byte length
// travels in the Inference-Header-Content-Length HTTP header; each binary
// tensor carries parameters.binary_data_size (and omits the inline "data"
// field), and tensor blobs are concatenated in tensor order after the JSON.
// This avoids the ~6x bloat of JSON number arrays for large tensors. The
// helpers below are pure so the framing is unit-testable without a socket.

// HTTP header name carrying the JSON-header byte length for binary payloads.
inline constexpr std::string_view kInferenceHeaderContentLength =
    "Inference-Header-Content-Length";

// Builds the JSON node for one binary input (name/datatype/shape with
// parameters.binary_data_size = bytes.size() and no inline "data"), appending
// `bytes` to the shared output blob in tensor order. The caller pushes the
// returned node into request["inputs"] and sends `blob` after the JSON header.
nlohmann::json appendBinaryInput(const std::string &name,
                                 const std::string &datatype,
                                 const std::vector<int64_t> &shape,
                                 const std::vector<std::uint8_t> &bytes,
                                 std::vector<std::uint8_t> &blob);

// Builds a requested-output node asking the server to return `name` as binary
// (parameters.binary_data = true).
nlohmann::json requestBinaryOutput(const std::string &name);

// Returns parameters.binary_data_size from a tensor JSON node, or -1 when the
// node has no such parameter (i.e. it carries inline JSON "data" instead).
long binaryDataSize(const nlohmann::json &node);

// Splits a framed binary body into its JSON header and trailing binary blob
// using `header_len` (the Inference-Header-Content-Length value). Returns the
// parsed JSON header and sets `blob` to the remaining bytes. Throws when
// `header_len` exceeds the body or the header is not valid JSON.
nlohmann::json splitBinaryBody(const std::string &body, std::size_t header_len,
                               std::vector<std::uint8_t> &blob);

// Returns `size` bytes from `blob` starting at `offset`, bounds-checked. Throws
// std::runtime_error when the slice runs past the end of the blob.
std::vector<std::uint8_t> sliceBlob(const std::vector<std::uint8_t> &blob,
                                    std::size_t offset, std::size_t size);

} // namespace kserve
