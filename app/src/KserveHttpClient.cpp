#include "KserveHttpClient.hpp"

#include "KserveProtocol.hpp"
#include "KserveRetry.hpp"

#include <cctype>
#include <cstdlib>
#include <memory>
#include <netdb.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <utility>

#ifdef NEURIPLO_INFER_WITH_KSERVE_TLS
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#endif

using Json = nlohmann::json;

namespace kserve {

// Byte transport over an established socket. The request-building code below is
// transport-agnostic: it writes a request string and reads bytes back, whether
// they travel in plaintext or through a TLS session. Declared at namespace
// scope (not in the anonymous namespace) so the keep-alive holder below can own
// one without tripping -Wsubobject-linkage.
class Connection {
public:
  virtual ~Connection() = default;
  virtual void sendAll(const std::string &data) = 0;
  // Reads up to buf_size bytes into buf. Returns the number of bytes read
  // (> 0), 0 when the peer closed the connection, or < 0 on a transport error.
  virtual long recvSome(char *buf, std::size_t buf_size) = 0;
};

// Persistent keep-alive connection. Not thread-safe: the client assumes
// inference calls on a single HttpClient are serialized (the original design
// held no lock either). The transport is reused across requests; any I/O error
// drops it so the next round-trip — driven by the retry loop — reconnects.
struct HttpConnection {
  std::unique_ptr<Connection> transport;
  std::string host;
  int port = 0;
  bool tls = false;

  void drop() {
    transport.reset();
    host.clear();
    port = 0;
    tls = false;
  }
};

namespace {

void setTimeout(int fd, int timeout_ms) {
  struct timeval timeout {};
  timeout.tv_sec = timeout_ms / 1000;
  timeout.tv_usec = (timeout_ms % 1000) * 1000;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

// IPv4/IPv6-capable, thread-safe connect (replaces gethostbyname).
int connectSocket(const std::string &host, int port, int timeout_ms) {
  struct addrinfo hints {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo *result = nullptr;
  const std::string port_str = std::to_string(port);
  const int rc = ::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
  if (rc != 0) {
    throw std::runtime_error("host resolution failed: " + host + ": " +
                             ::gai_strerror(rc));
  }

  int fd = -1;
  for (struct addrinfo *ai = result; ai != nullptr; ai = ai->ai_next) {
    fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) {
      continue;
    }
    setTimeout(fd, timeout_ms);
    if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
      break;
    }
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(result);

  if (fd < 0) {
    throw std::runtime_error("connect failed to " + host + ":" +
                             std::to_string(port));
  }
  return fd;
}

class PlainConnection : public Connection {
public:
  explicit PlainConnection(int fd) : fd_(fd) {}
  ~PlainConnection() override {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }
  PlainConnection(const PlainConnection &) = delete;
  PlainConnection &operator=(const PlainConnection &) = delete;

  void sendAll(const std::string &data) override {
    size_t sent = 0;
    while (sent < data.size()) {
      const auto n = ::send(fd_, data.data() + sent, data.size() - sent, 0);
      if (n <= 0) {
        throw std::runtime_error("HTTP send failed");
      }
      sent += static_cast<size_t>(n);
    }
  }

  long recvSome(char *buf, std::size_t buf_size) override {
    const auto received = ::recv(fd_, buf, buf_size, 0);
    return static_cast<long>(received);
  }

private:
  int fd_;
};

#ifdef NEURIPLO_INFER_WITH_KSERVE_TLS
// TLS transport built on OpenSSL. Verifies the server certificate against the
// system CA roots (or KSERVE_CA_CERT when set), sends SNI, and checks the
// certificate hostname against the endpoint host.
class TlsConnection : public Connection {
public:
  TlsConnection(int fd, const std::string &host) : fd_(fd) {
    ctx_ = SSL_CTX_new(TLS_client_method());
    if (ctx_ == nullptr) {
      fail("SSL_CTX_new failed");
    }
    SSL_CTX_set_min_proto_version(ctx_, TLS1_2_VERSION);
    SSL_CTX_set_verify(ctx_, SSL_VERIFY_PEER, nullptr);

    const char *ca_path = std::getenv("KSERVE_CA_CERT");
    if (ca_path != nullptr && ca_path[0] != '\0') {
      if (SSL_CTX_load_verify_locations(ctx_, ca_path, nullptr) != 1) {
        fail("failed to load CA file named by KSERVE_CA_CERT");
      }
    } else if (SSL_CTX_set_default_verify_paths(ctx_) != 1) {
      fail("failed to load system CA roots");
    }

    ssl_ = SSL_new(ctx_);
    if (ssl_ == nullptr) {
      fail("SSL_new failed");
    }
    SSL_set_fd(ssl_, fd_);
    // SNI: tell the server which virtual host we want.
    SSL_set_tlsext_host_name(ssl_, host.c_str());
    // Verify the certificate matches the host we connected to.
    X509_VERIFY_PARAM *param = SSL_get0_param(ssl_);
    X509_VERIFY_PARAM_set_hostflags(param,
                                    X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
    if (X509_VERIFY_PARAM_set1_host(param, host.c_str(), host.size()) != 1) {
      fail("failed to set expected TLS hostname");
    }
    if (SSL_connect(ssl_) != 1) {
      fail("TLS handshake failed");
    }
  }

  ~TlsConnection() override { cleanup(); }
  TlsConnection(const TlsConnection &) = delete;
  TlsConnection &operator=(const TlsConnection &) = delete;

  void sendAll(const std::string &data) override {
    size_t sent = 0;
    while (sent < data.size()) {
      const int n = SSL_write(ssl_, data.data() + sent,
                              static_cast<int>(data.size() - sent));
      if (n <= 0) {
        throw std::runtime_error("TLS send failed");
      }
      sent += static_cast<size_t>(n);
    }
  }

  long recvSome(char *buf, std::size_t buf_size) override {
    const int n = SSL_read(ssl_, buf, static_cast<int>(buf_size));
    // Treat any non-positive return as a clean end-of-response; the framing
    // logic decides completeness from the bytes already read.
    return n > 0 ? static_cast<long>(n) : 0;
  }

private:
  void cleanup() {
    if (ssl_ != nullptr) {
      SSL_shutdown(ssl_);
      SSL_free(ssl_);
      ssl_ = nullptr;
    }
    if (ctx_ != nullptr) {
      SSL_CTX_free(ctx_);
      ctx_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  // Captures the OpenSSL error string, releases all resources, and throws.
  [[noreturn]] void fail(const std::string &what) {
    const unsigned long err = ERR_get_error();
    std::string message = "KServe TLS: " + what;
    if (err != 0) {
      char buf[256];
      ERR_error_string_n(err, buf, sizeof(buf));
      message += ": ";
      message += buf;
    }
    cleanup();
    throw std::runtime_error(message);
  }

  int fd_;
  SSL_CTX *ctx_ = nullptr;
  SSL *ssl_ = nullptr;
};
#endif // NEURIPLO_INFER_WITH_KSERVE_TLS

// Establishes a connection to the endpoint, wrapping it in TLS for https://.
// A TLS endpoint on a build without OpenSSL fails fast with a clear message.
std::unique_ptr<Connection> connect(const Endpoint &ep, int timeout_ms) {
  const int fd = connectSocket(ep.host, ep.port, timeout_ms);
  if (ep.tls) {
#ifdef NEURIPLO_INFER_WITH_KSERVE_TLS
    return std::make_unique<TlsConnection>(fd, ep.host);
#else
    ::close(fd);
    throw std::runtime_error(
        "KServe https:// endpoint requested but the client was built without "
        "TLS support (rebuild with OpenSSL / "
        "-DNEURIPLO_INFER_ENABLE_KSERVE_TLS=ON)");
#endif
  }
  return std::make_unique<PlainConnection>(fd);
}

std::string toLower(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

// Decides whether `raw` already holds a complete HTTP/1.1 message, so the
// framed reader can stop without relying on EOF (required for keep-alive).
// Honours Content-Length and Transfer-Encoding: chunked; falls back to
// "complete only at EOF" for close-delimited responses.
bool messageComplete(const std::string &raw) {
  const auto header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    return false;
  }
  const std::string headers = toLower(raw.substr(0, header_end));
  const std::size_t body_start = header_end + 4;

  if (headers.find("transfer-encoding:") != std::string::npos &&
      headers.find("chunked") != std::string::npos) {
    // Terminated by the zero-length chunk "0\r\n\r\n".
    return raw.find("\r\n0\r\n\r\n", header_end) != std::string::npos ||
           raw.compare(body_start, 5, "0\r\n\r\n") == 0;
  }

  // Match Content-Length at a header-line boundary so it is not confused with
  // the Inference-Header-Content-Length header, which ends in the same text.
  const std::string marker = "\r\ncontent-length:";
  const auto pos = headers.find(marker);
  if (pos != std::string::npos) {
    try {
      const long len = std::stol(headers.substr(pos + marker.size()));
      return raw.size() >= body_start + static_cast<std::size_t>(len);
    } catch (...) {
      return false;
    }
  }
  // No framing information: close-delimited, only complete at EOF.
  return false;
}

// Reads one framed HTTP response over `transport`. Stops as soon as the message
// is complete (so the connection can be kept alive) or at EOF for
// close-delimited responses. Sets `server_closed` when the peer closed first.
std::string readFramed(Connection &transport, bool &server_closed) {
  server_closed = false;
  std::string raw;
  char buffer[4096];
  while (!messageComplete(raw)) {
    const long received = transport.recvSome(buffer, sizeof(buffer));
    if (received < 0) {
      throw std::runtime_error("HTTP recv failed");
    }
    if (received == 0) {
      server_closed = true;
      break;
    }
    raw.append(buffer, static_cast<std::size_t>(received));
  }
  if (raw.empty()) {
    throw std::runtime_error("HTTP read failed: empty response");
  }
  return raw;
}

// True when the response asked to close the connection (HTTP/1.0 default, or an
// explicit "Connection: close").
bool wantsClose(const std::string &raw) {
  const auto header_end = raw.find("\r\n\r\n");
  const std::string headers = toLower(
      header_end == std::string::npos ? raw : raw.substr(0, header_end));
  return headers.find("connection: close") != std::string::npos;
}

// Returns the live transport for `ep`, reusing the persistent connection when
// it already targets the same host/port/scheme, otherwise (re)connecting.
Connection &ensureConnected(HttpConnection &conn, const Endpoint &ep,
                            int timeout_ms) {
  if (conn.transport && conn.host == ep.host && conn.port == ep.port &&
      conn.tls == ep.tls) {
    return *conn.transport;
  }
  conn.drop();
  conn.transport = connect(ep, timeout_ms);
  conn.host = ep.host;
  conn.port = ep.port;
  conn.tls = ep.tls;
  return *conn.transport;
}

// Performs a single keep-alive round-trip: (re)connect if needed, send, read
// the framed response, and parse it. Any I/O failure drops the connection and
// rethrows so the retry loop reconnects on the next attempt.
HttpResponse roundTrip(HttpConnection &conn, const Endpoint &ep,
                       const std::string &request, int timeout_ms) {
  try {
    Connection &transport = ensureConnected(conn, ep, timeout_ms);
    transport.sendAll(request);
    bool server_closed = false;
    const std::string raw = readFramed(transport, server_closed);
    auto response = parseHttpResponse(raw);
    if (server_closed || wantsClose(raw)) {
      conn.drop();
    }
    return response;
  } catch (...) {
    conn.drop();
    throw;
  }
}

std::string buildRequest(const char *method, const Endpoint &ep,
                         const std::string &path, const std::string &auth_token,
                         const std::string &content_type,
                         const std::string &extra_headers,
                         const std::string &body) {
  std::ostringstream request;
  request << method << " " << path << " HTTP/1.1\r\n"
          << "Host: " << ep.host << "\r\n"
          << "Connection: keep-alive\r\n";
  if (!content_type.empty()) {
    request << "Content-Type: " << content_type << "\r\n";
  }
  if (!auth_token.empty()) {
    request << "Authorization: Bearer " << auth_token << "\r\n";
  }
  request << extra_headers;
  // Always frame the body with Content-Length so keep-alive readers know the
  // exact length (zero-length for GET).
  request << "Content-Length: " << body.size() << "\r\n\r\n" << body;
  return request.str();
}

// Wraps a single round-trip in the retry/backoff loop: transport exceptions and
// retryable HTTP statuses (429/502/503/504) are retried; everything else (and
// the final attempt) surfaces unchanged.
template <typename Attempt>
HttpResponse withRetry(const RetryPolicy &policy, Attempt attempt) {
  return runWithRetry(
      policy, attempt,
      [](const HttpResponse &r) { return isRetryableHttpStatus(r.status); },
      defaultSleep, defaultJitter);
}

std::string inferPath(const std::string &model_name,
                      const std::string &model_version) {
  if (model_version.empty()) {
    return "/v2/models/" + model_name + "/infer";
  }
  return "/v2/models/" + model_name + "/versions/" + model_version + "/infer";
}

std::string envToken() {
  return readSecretFromEnvOrFile("KSERVE_BEARER_TOKEN",
                                 "KSERVE_BEARER_TOKEN_FILE");
}

// HTTP binary tensor extension is opt-in (the JSON path is the validated
// default): enable with a truthy KSERVE_BINARY (1/true/on/yes).
bool binaryEnabled() {
  const char *raw = std::getenv("KSERVE_BINARY");
  if (raw == nullptr) {
    return false;
  }
  const std::string value = toLower(raw);
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

std::vector<int64_t> readShape(const Json &node) {
  std::vector<int64_t> shape;
  if (node.contains("shape")) {
    for (const auto &dim : node["shape"]) {
      shape.push_back(dim.get<int64_t>());
    }
  }
  return shape;
}

} // namespace

HttpClient::HttpClient(std::string endpoint, std::string model_name,
                       std::string model_version, int timeout_ms)
    : endpoint_(std::move(endpoint)), model_name_(std::move(model_name)),
      model_version_(std::move(model_version)), timeout_ms_(timeout_ms),
      auth_token_(envToken()), retry_policy_(retryPolicyFromEnv()),
      conn_(std::make_shared<HttpConnection>()) {}

HttpClient::~HttpClient() = default;

ModelMetadata HttpClient::modelMetadata() {
  const auto ep = parseEndpoint(endpoint_, 8080);
  const std::string path = ep.path_prefix + "/v2/models/" + model_name_;
  const std::string request =
      buildRequest("GET", ep, path, auth_token_, "", "", "");

  const auto http_response = withRetry(retry_policy_, [&] {
    return roundTrip(*conn_, ep, request, timeout_ms_);
  });
  if (http_response.status != 200) {
    throw std::runtime_error("KServe metadata fetch failed: HTTP " +
                             std::to_string(http_response.status));
  }

  auto json = Json::parse(http_response.body);
  if (!json.contains("inputs") || !json.contains("outputs")) {
    throw std::runtime_error("KServe metadata response missing inputs/outputs");
  }

  ModelMetadata metadata;
  for (const auto &input : json["inputs"]) {
    metadata.inputs.push_back({input["name"].get<std::string>(),
                               input.value("datatype", "FP32"),
                               readShape(input)});
  }
  output_names_.clear();
  for (const auto &output : json["outputs"]) {
    metadata.outputs.push_back({output["name"].get<std::string>(),
                                output.value("datatype", "FP32"),
                                readShape(output)});
    output_names_.push_back(output["name"].get<std::string>());
  }
  return metadata;
}

bool HttpClient::probe(const std::string &path) {
  const auto ep = parseEndpoint(endpoint_, 8080);
  const std::string request =
      buildRequest("GET", ep, ep.path_prefix + path, auth_token_, "", "", "");
  return withRetry(retry_policy_, [&] {
           return roundTrip(*conn_, ep, request, timeout_ms_);
         }).status == 200;
}

bool HttpClient::serverLive() { return probe(serverLivePath()); }

bool HttpClient::serverReady() { return probe(serverReadyPath()); }

bool HttpClient::modelReady() {
  return probe(modelReadyPath(model_name_, model_version_));
}

std::vector<InferOutput>
HttpClient::infer(const std::vector<InferInput> &inputs) {
  const auto ep = parseEndpoint(endpoint_, 8080);
  const std::string path =
      ep.path_prefix + inferPath(model_name_, model_version_);
  const bool binary = binaryEnabled();

  Json request;
  request["id"] = "kserve-client";
  Json json_inputs = Json::array();
  std::vector<std::uint8_t> blob; // binary tensor section (binary mode only)
  for (const auto &input : inputs) {
    if (input.data == nullptr) {
      throw std::runtime_error("KServe input '" + input.name + "' has no data");
    }
    if (binary) {
      json_inputs.push_back(appendBinaryInput(input.name, input.datatype,
                                              input.shape, *input.data, blob));
    } else {
      Json node;
      node["name"] = input.name;
      node["datatype"] = input.datatype;
      node["shape"] = input.shape;
      node["data"] = encodeTensorData(*input.data, input.datatype);
      json_inputs.push_back(std::move(node));
    }
  }
  request["inputs"] = std::move(json_inputs);

  // Request binary outputs only when we know their names (from a prior
  // modelMetadata() call) so the server frames them as raw bytes too.
  if (binary && !output_names_.empty()) {
    Json json_outputs = Json::array();
    for (const auto &name : output_names_) {
      json_outputs.push_back(requestBinaryOutput(name));
    }
    request["outputs"] = std::move(json_outputs);
  }

  std::string content_type = "application/json";
  std::string extra_headers;
  std::string body = request.dump();
  if (binary) {
    content_type = "application/octet-stream";
    extra_headers = std::string(kInferenceHeaderContentLength) + ": " +
                    std::to_string(body.size()) + "\r\n";
    body.append(reinterpret_cast<const char *>(blob.data()), blob.size());
  }

  const std::string http_request = buildRequest(
      "POST", ep, path, auth_token_, content_type, extra_headers, body);
  const auto http_response = withRetry(retry_policy_, [&] {
    return roundTrip(*conn_, ep, http_request, timeout_ms_);
  });

  if (http_response.status != 200) {
    throw std::runtime_error("KServe inference failed: HTTP " +
                             std::to_string(http_response.status) + " " +
                             http_response.body);
  }

  // Binary response: JSON header followed by the raw tensor blob.
  if (http_response.inference_header_length >= 0) {
    std::vector<std::uint8_t> out_blob;
    auto header = splitBinaryBody(
        http_response.body,
        static_cast<std::size_t>(http_response.inference_header_length),
        out_blob);
    if (!header.contains("outputs") || !header["outputs"].is_array()) {
      throw std::runtime_error("KServe response missing outputs");
    }
    std::vector<InferOutput> outputs;
    std::size_t offset = 0;
    for (const auto &output : header["outputs"]) {
      InferOutput out;
      out.name = output.value("name", "");
      out.datatype = output.value("datatype", "FP32");
      out.shape = readShape(output);
      const long size = binaryDataSize(output);
      if (size >= 0) {
        out.data = sliceBlob(out_blob, offset, static_cast<std::size_t>(size));
        offset += static_cast<std::size_t>(size);
      } else if (output.contains("data") && output["data"].is_array()) {
        out.data = decodeTensorData(output["data"], out.datatype);
      }
      outputs.push_back(std::move(out));
    }
    return outputs;
  }

  auto json_response = Json::parse(http_response.body);
  if (!json_response.contains("outputs") ||
      !json_response["outputs"].is_array()) {
    throw std::runtime_error("KServe response missing outputs");
  }

  std::vector<InferOutput> outputs;
  for (const auto &output : json_response["outputs"]) {
    InferOutput out;
    out.name = output.value("name", "");
    out.datatype = output.value("datatype", "FP32");
    out.shape = readShape(output);
    if (output.contains("data") && output["data"].is_array()) {
      out.data = decodeTensorData(output["data"], out.datatype);
    }
    outputs.push_back(std::move(out));
  }
  return outputs;
}

} // namespace kserve
