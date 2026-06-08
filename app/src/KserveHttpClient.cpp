#include "KserveHttpClient.hpp"

#include "KserveProtocol.hpp"

#include <cstdlib>
#include <memory>
#include <netdb.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
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

namespace {

void setTimeout(int fd, int timeout_ms) {
  struct timeval timeout;
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

// Byte transport over an established socket. The request-building code below is
// transport-agnostic: it writes a request string and reads the raw response,
// whether the bytes travel in plaintext or through a TLS session. Each
// connection owns its fd and closes it on destruction.
class Connection {
public:
  virtual ~Connection() = default;
  virtual void sendAll(const std::string &data) = 0;
  virtual std::string readAll() = 0;
};

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

  std::string readAll() override {
    char buffer[4096];
    std::string raw;
    while (true) {
      const auto received = ::recv(fd_, buffer, sizeof(buffer), 0);
      if (received <= 0) {
        break;
      }
      raw.append(buffer, static_cast<size_t>(received));
    }
    return raw;
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

  std::string readAll() override {
    char buffer[4096];
    std::string raw;
    while (true) {
      const int n = SSL_read(ssl_, buffer, sizeof(buffer));
      if (n <= 0) {
        break;
      }
      raw.append(buffer, static_cast<size_t>(n));
    }
    return raw;
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

HttpResponse httpGet(const Endpoint &ep, const std::string &path,
                     const std::string &auth_token, int timeout_ms) {
  auto conn = connect(ep, timeout_ms);
  std::ostringstream request;
  request << "GET " << path << " HTTP/1.1\r\n"
          << "Host: " << ep.host << "\r\n";
  if (!auth_token.empty()) {
    request << "Authorization: Bearer " << auth_token << "\r\n";
  }
  request << "Connection: close\r\n\r\n";
  conn->sendAll(request.str());
  return parseHttpResponse(conn->readAll());
}

HttpResponse httpPost(const Endpoint &ep, const std::string &path,
                      const std::string &body, const std::string &auth_token,
                      int timeout_ms) {
  auto conn = connect(ep, timeout_ms);
  std::ostringstream request;
  request << "POST " << path << " HTTP/1.1\r\n"
          << "Host: " << ep.host << "\r\n"
          << "Content-Type: application/json\r\n"
          << "Content-Length: " << body.size() << "\r\n";
  if (!auth_token.empty()) {
    request << "Authorization: Bearer " << auth_token << "\r\n";
  }
  request << "Connection: close\r\n\r\n" << body;
  conn->sendAll(request.str());
  return parseHttpResponse(conn->readAll());
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
      auth_token_(envToken()) {}

ModelMetadata HttpClient::modelMetadata() {
  const auto ep = parseEndpoint(endpoint_, 8080);
  const std::string path = ep.path_prefix + "/v2/models/" + model_name_;

  const auto http_response = httpGet(ep, path, auth_token_, timeout_ms_);
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
  for (const auto &output : json["outputs"]) {
    metadata.outputs.push_back({output["name"].get<std::string>(),
                                output.value("datatype", "FP32"),
                                readShape(output)});
  }
  return metadata;
}

bool HttpClient::probe(const std::string &path) {
  const auto ep = parseEndpoint(endpoint_, 8080);
  return httpGet(ep, ep.path_prefix + path, auth_token_, timeout_ms_).status ==
         200;
}

bool HttpClient::serverLive() { return probe(serverLivePath()); }

bool HttpClient::serverReady() { return probe(serverReadyPath()); }

bool HttpClient::modelReady() {
  return probe(modelReadyPath(model_name_, model_version_));
}

std::vector<InferOutput>
HttpClient::infer(const std::vector<InferInput> &inputs) {
  const auto ep = parseEndpoint(endpoint_, 8080);

  Json request;
  request["id"] = "kserve-client";
  Json json_inputs = Json::array();
  for (const auto &input : inputs) {
    if (input.data == nullptr) {
      throw std::runtime_error("KServe input '" + input.name + "' has no data");
    }
    Json node;
    node["name"] = input.name;
    node["datatype"] = input.datatype;
    node["shape"] = input.shape;
    node["data"] = encodeTensorData(*input.data, input.datatype);
    json_inputs.push_back(std::move(node));
  }
  request["inputs"] = std::move(json_inputs);

  const auto http_response =
      httpPost(ep, ep.path_prefix + inferPath(model_name_, model_version_),
               request.dump(), auth_token_, timeout_ms_);

  if (http_response.status != 200) {
    throw std::runtime_error("KServe inference failed: HTTP " +
                             std::to_string(http_response.status) + " " +
                             http_response.body);
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
