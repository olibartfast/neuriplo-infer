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

// --- Secret / TLS configuration helpers --------------------------------------
// Secrets and TLS material are always sourced from environment variables or
// files they name, never from the command line. The helpers below split the
// pure resolution logic (unit-testable without env/filesystem) from the impure
// readers that touch the environment and disk.

// Removes trailing whitespace (spaces, tabs, CR, LF) from a value read from a
// file, so a trailing newline does not leak into a token. Pure.
std::string trimTrailingWhitespace(const std::string &value);

// Resolves a secret with env-over-file precedence: a non-empty env value is
// returned verbatim; otherwise the file contents (if any) are returned with
// trailing whitespace trimmed. Either pointer may be null (= unset). Returns
// empty when neither source yields a value. Pure.
std::string resolveSecret(const std::string *env_value,
                          const std::string *file_contents);

// Validates the mTLS client cert/key pairing: both must be present or both
// absent. Returns true when a client certificate pair is configured; throws
// std::runtime_error when only one of cert/key is provided. Pure.
bool requireClientCertPair(const std::string &client_cert,
                           const std::string &client_key);

// Reads an entire file into a string. Throws std::runtime_error if the file
// cannot be opened. Impure (filesystem).
std::string readFileToString(const std::string &path);

// Reads a secret from `env_var`, falling back to the file named by
// `file_env_var` when the env var is unset/empty (trailing whitespace trimmed).
// Returns empty when neither is set. Impure (env + filesystem); the pure
// resolution logic lives in resolveSecret / trimTrailingWhitespace.
std::string readSecretFromEnvOrFile(const char *env_var,
                                    const char *file_env_var);

// Returns the contents of the file whose path is held in `env_var`, or an empty
// string when the env var is unset/empty. Used for PEM CA/cert/key material.
// Impure (env + filesystem).
std::string readFileFromEnvPath(const char *env_var);

} // namespace kserve
