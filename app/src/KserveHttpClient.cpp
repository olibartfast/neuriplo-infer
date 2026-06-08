#include "KserveHttpClient.hpp"

#include "KserveProtocol.hpp"

#include <cstdlib>
#include <netdb.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <utility>

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

void sendAll(int fd, const std::string &request) {
  size_t sent = 0;
  while (sent < request.size()) {
    const auto n = ::send(fd, request.data() + sent, request.size() - sent, 0);
    if (n <= 0) {
      throw std::runtime_error("HTTP send failed");
    }
    sent += static_cast<size_t>(n);
  }
}

std::string readRaw(int fd) {
  char buffer[4096];
  std::string raw;
  while (true) {
    const auto received = ::recv(fd, buffer, sizeof(buffer), 0);
    if (received <= 0) {
      break;
    }
    raw.append(buffer, static_cast<size_t>(received));
  }
  return raw;
}

HttpResponse httpGet(const Endpoint &ep, const std::string &path,
                     const std::string &auth_token, int timeout_ms) {
  const int fd = connectSocket(ep.host, ep.port, timeout_ms);
  try {
    std::ostringstream request;
    request << "GET " << path << " HTTP/1.1\r\n"
            << "Host: " << ep.host << "\r\n";
    if (!auth_token.empty()) {
      request << "Authorization: Bearer " << auth_token << "\r\n";
    }
    request << "Connection: close\r\n\r\n";
    sendAll(fd, request.str());
    auto response = parseHttpResponse(readRaw(fd));
    ::close(fd);
    return response;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

HttpResponse httpPost(const Endpoint &ep, const std::string &path,
                      const std::string &body, const std::string &auth_token,
                      int timeout_ms) {
  const int fd = connectSocket(ep.host, ep.port, timeout_ms);
  try {
    std::ostringstream request;
    request << "POST " << path << " HTTP/1.1\r\n"
            << "Host: " << ep.host << "\r\n"
            << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n";
    if (!auth_token.empty()) {
      request << "Authorization: Bearer " << auth_token << "\r\n";
    }
    request << "Connection: close\r\n\r\n" << body;
    sendAll(fd, request.str());
    auto response = parseHttpResponse(readRaw(fd));
    ::close(fd);
    return response;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

std::string inferPath(const std::string &model_name,
                      const std::string &model_version) {
  if (model_version.empty()) {
    return "/v2/models/" + model_name + "/infer";
  }
  return "/v2/models/" + model_name + "/versions/" + model_version + "/infer";
}

std::string envToken() {
  const char *token = std::getenv("KSERVE_BEARER_TOKEN");
  return token != nullptr ? std::string(token) : std::string();
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
