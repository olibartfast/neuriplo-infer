#include "KserveClient.hpp"

#include <arpa/inet.h>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

using Json = nlohmann::json;

namespace {

struct HttpResponse {
  int status = 0;
  std::string body;
};

std::string stripScheme(const std::string &url) {
  if (url.rfind("http://", 0) == 0) {
    return url.substr(7);
  }
  if (url.rfind("https://", 0) == 0) {
    return url.substr(8);
  }
  return url;
}

std::string authority(const std::string &url) {
  auto value = stripScheme(url);
  const auto slash = value.find('/');
  if (slash != std::string::npos) {
    value = value.substr(0, slash);
  }
  return value;
}

std::string extractHost(const std::string &url) {
  auto host = authority(url);
  const auto colon = host.find(':');
  if (colon != std::string::npos) {
    host = host.substr(0, colon);
  }
  return host;
}

int extractPort(const std::string &url, int default_port) {
  const auto host = authority(url);
  const auto colon = host.find(':');
  if (colon != std::string::npos) {
    return std::stoi(host.substr(colon + 1));
  }
  return default_port;
}

std::string extractPathPrefix(const std::string &url) {
  const auto value = stripScheme(url);
  const auto slash = value.find('/');
  if (slash == std::string::npos) {
    return "";
  }
  auto prefix = value.substr(slash);
  while (!prefix.empty() && prefix.back() == '/') {
    prefix.pop_back();
  }
  return prefix;
}

void setTimeout(int fd, int timeout_ms) {
  struct timeval timeout;
  timeout.tv_sec = timeout_ms / 1000;
  timeout.tv_usec = (timeout_ms % 1000) * 1000;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

int connectSocket(const std::string &host, int port, int timeout_ms) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw std::runtime_error("socket failed");
  }
  setTimeout(fd, timeout_ms);

  struct hostent *server = ::gethostbyname(host.c_str());
  if (server == nullptr) {
    ::close(fd);
    throw std::runtime_error("host resolution failed: " + host);
  }

  struct sockaddr_in address {};
  address.sin_family = AF_INET;
  std::memcpy(&address.sin_addr.s_addr, server->h_addr,
              static_cast<size_t>(server->h_length));
  address.sin_port = htons(static_cast<uint16_t>(port));

  if (::connect(fd, reinterpret_cast<struct sockaddr *>(&address),
                sizeof(address)) < 0) {
    ::close(fd);
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

HttpResponse readResponse(int fd) {
  HttpResponse response;
  char buffer[4096];
  std::string raw;
  while (true) {
    const auto received = ::recv(fd, buffer, sizeof(buffer), 0);
    if (received <= 0) {
      break;
    }
    raw.append(buffer, static_cast<size_t>(received));
  }

  const auto first_space = raw.find(' ');
  if (first_space == std::string::npos) {
    throw std::runtime_error("invalid HTTP response");
  }
  const auto second_space = raw.find(' ', first_space + 1);
  response.status =
      std::stoi(raw.substr(first_space + 1, second_space - first_space - 1));

  const auto body_start = raw.find("\r\n\r\n");
  if (body_start != std::string::npos) {
    response.body = raw.substr(body_start + 4);
  }
  return response;
}

HttpResponse httpGet(const std::string &host, int port, const std::string &path,
                     int timeout_ms) {
  const int fd = connectSocket(host, port, timeout_ms);
  try {
    std::ostringstream request;
    request << "GET " << path << " HTTP/1.1\r\n"
            << "Host: " << host << "\r\n"
            << "Connection: close\r\n"
            << "\r\n";
    sendAll(fd, request.str());
    auto response = readResponse(fd);
    ::close(fd);
    return response;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

HttpResponse httpPost(const std::string &host, int port,
                      const std::string &path, const std::string &body,
                      int timeout_ms) {
  const int fd = connectSocket(host, port, timeout_ms);
  try {
    std::ostringstream request;
    request << "POST " << path << " HTTP/1.1\r\n"
            << "Host: " << host << "\r\n"
            << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Connection: close\r\n"
            << "\r\n"
            << body;
    sendAll(fd, request.str());
    auto response = readResponse(fd);
    ::close(fd);
    return response;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

size_t batchSize(const std::vector<int64_t> &shape) {
  if (!shape.empty() && shape[0] > 0) {
    return static_cast<size_t>(shape[0]);
  }
  return 1;
}

bool parseMetadataJson(const std::string &body, InferenceMetadata &meta) {
  auto json = Json::parse(body);
  if (!json.contains("inputs") || !json.contains("outputs")) {
    return false;
  }
  for (const auto &input : json["inputs"]) {
    std::vector<int64_t> shape;
    for (const auto &dim : input["shape"]) {
      shape.push_back(dim.get<int64_t>());
    }
    meta.addInput(input["name"].get<std::string>(), shape, batchSize(shape));
  }
  for (const auto &output : json["outputs"]) {
    std::vector<int64_t> shape;
    for (const auto &dim : output["shape"]) {
      shape.push_back(dim.get<int64_t>());
    }
    meta.addOutput(output["name"].get<std::string>(), shape, batchSize(shape));
  }
  return true;
}

Json fp32BytesToJson(const std::vector<uint8_t> &bytes) {
  if (bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("FP32 input byte count is not divisible by 4");
  }
  Json data = Json::array();
  for (size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
    float value = 0.0F;
    std::memcpy(&value, bytes.data() + offset, sizeof(float));
    data.push_back(value);
  }
  return data;
}

TensorElement parseTensorElement(const Json &value,
                                 const std::string &datatype) {
  if (datatype == "FP32") {
    return static_cast<float>(value.get<double>());
  }
  if (datatype == "INT32") {
    return static_cast<int32_t>(value.get<int32_t>());
  }
  if (datatype == "INT64") {
    return static_cast<int64_t>(value.get<int64_t>());
  }
  if (datatype == "UINT8") {
    return static_cast<uint8_t>(value.get<unsigned int>());
  }
  throw std::runtime_error("unsupported KServe output datatype: " + datatype);
}

std::string inferPath(const std::string &model_name,
                      const std::string &model_version) {
  if (model_version.empty()) {
    return "/v2/models/" + model_name + "/infer";
  }
  return "/v2/models/" + model_name + "/versions/" + model_version + "/infer";
}

} // namespace

KserveClient::KserveClient(const std::string &endpoint,
                           const std::string &model_name,
                           const std::string &model_version, int timeout_ms)
    : InferenceInterface("", false, 1), endpoint_(endpoint),
      model_name_(model_name), model_version_(model_version),
      timeout_ms_(timeout_ms) {}

std::tuple<std::vector<std::vector<TensorElement>>,
           std::vector<std::vector<int64_t>>>
KserveClient::get_infer_results(
    const std::vector<std::vector<uint8_t>> &input_tensors) {
  if (!metadata_loaded_) {
    throw std::runtime_error("KServe client metadata not loaded");
  }

  const auto host = extractHost(endpoint_);
  const auto port = extractPort(endpoint_, 8080);
  const auto prefix = extractPathPrefix(endpoint_);

  Json request;
  request["id"] = "kserve-client";
  Json inputs = Json::array();

  const auto &meta_inputs = cached_metadata_.getInputs();
  if (input_tensors.size() != meta_inputs.size()) {
    throw std::runtime_error("KServe input tensor count mismatch");
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    Json input;
    input["name"] = meta_inputs[i].name;
    input["datatype"] = "FP32";
    input["shape"] = meta_inputs[i].shape;
    input["data"] = fp32BytesToJson(input_tensors[i]);
    inputs.push_back(input);
  }
  request["inputs"] = inputs;

  Json outputs = Json::array();
  const auto &meta_outputs = cached_metadata_.getOutputs();
  for (const auto &output : meta_outputs) {
    Json out;
    out["name"] = output.name;
    outputs.push_back(out);
  }
  request["outputs"] = outputs;

  const auto http_response =
      httpPost(host, port, prefix + inferPath(model_name_, model_version_),
               request.dump(), timeout_ms_);

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

  std::vector<std::vector<TensorElement>> output_data;
  std::vector<std::vector<int64_t>> output_shapes;

  for (const auto &output : json_response["outputs"]) {
    std::vector<int64_t> shape;
    if (output.contains("shape")) {
      for (const auto &dim : output["shape"]) {
        shape.push_back(dim.get<int64_t>());
      }
    }
    output_shapes.push_back(shape);

    const auto datatype = output.value("datatype", "FP32");
    std::vector<TensorElement> data;
    if (output.contains("data") && output["data"].is_array()) {
      for (const auto &value : output["data"]) {
        data.push_back(parseTensorElement(value, datatype));
      }
    }
    output_data.push_back(std::move(data));
  }

  return {std::move(output_data), std::move(output_shapes)};
}

InferenceMetadata KserveClient::get_inference_metadata() {
  if (!metadata_loaded_) {
    fetchMetadata();
  }
  return cached_metadata_;
}

bool KserveClient::is_gpu_available() const noexcept { return false; }

bool KserveClient::fetchMetadata() {
  const auto host = extractHost(endpoint_);
  const auto port = extractPort(endpoint_, 8080);
  const auto prefix = extractPathPrefix(endpoint_);
  const std::string path = prefix + "/v2/models/" + model_name_;

  const auto http_response = httpGet(host, port, path, timeout_ms_);

  if (http_response.status != 200) {
    throw std::runtime_error("KServe metadata fetch failed: HTTP " +
                             std::to_string(http_response.status));
  }

  metadata_loaded_ = parseMetadataJson(http_response.body, cached_metadata_);
  return metadata_loaded_;
}
