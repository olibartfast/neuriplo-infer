#pragma once

// Pure, dependency-light helpers for the KServe V2 / Open Inference Protocol.
// Nothing here depends on the neuriplo backend types, so it is unit-testable in
// isolation (only nlohmann/json + the standard library).

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
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
};

// Parses a raw HTTP/1.1 response (headers + body). Honours Content-Length and
// Transfer-Encoding: chunked. Throws std::runtime_error on a malformed response.
HttpResponse parseHttpResponse(const std::string &raw);

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
// packs it into raw little-endian bytes for `datatype`. FP16 values are narrowed
// back to half. Throws on a non-array input or an unsupported datatype.
std::vector<std::uint8_t> decodeTensorData(const nlohmann::json &data,
                                           const std::string &datatype);

} // namespace kserve
