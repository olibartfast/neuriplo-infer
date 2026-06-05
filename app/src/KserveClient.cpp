#include "KserveClient.hpp"

#include <arpa/inet.h>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

using Json = nlohmann::json;

namespace {

struct HttpResponse {
    int status = 0;
    std::string body;
};

std::string extractHost(const std::string &url) {
    auto host = url;
    if (host.rfind("http://", 0) == 0) {
        host = host.substr(7);
    } else if (host.rfind("https://", 0) == 0) {
        host = host.substr(8);
    }
    const auto colon = host.find(':');
    if (colon != std::string::npos) {
        host = host.substr(0, colon);
    }
    return host;
}

int extractPort(const std::string &url, int default_port) {
    auto host = url;
    if (host.rfind("http://", 0) == 0) {
        host = host.substr(7);
    } else if (host.rfind("https://", 0) == 0) {
        host = host.substr(8);
    }
    const auto slash = host.find('/');
    if (slash != std::string::npos) {
        host = host.substr(0, slash);
    }
    const auto colon = host.find(':');
    if (colon != std::string::npos) {
        return std::stoi(host.substr(colon + 1));
    }
    return default_port;
}

HttpResponse httpGet(const std::string &host, int port, const std::string &path,
                     int timeout_ms) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }

    struct timeval timeout;
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    struct hostent *server = ::gethostbyname(host.c_str());
    if (server == nullptr) {
        ::close(fd);
        throw std::runtime_error("host resolution failed: " + host);
    }

    struct sockaddr_in address {};
    address.sin_family = AF_INET;
    std::memcpy(&address.sin_addr.s_addr, server->h_addr, static_cast<size_t>(server->h_length));
    address.sin_port = htons(static_cast<uint16_t>(port));

    if (::connect(fd, reinterpret_cast<struct sockaddr *>(&address), sizeof(address)) < 0) {
        ::close(fd);
        throw std::runtime_error("connect failed to " + host + ":" + std::to_string(port));
    }

    std::ostringstream request;
    request << "GET " << path << " HTTP/1.1\r\n"
            << "Host: " << host << "\r\n"
            << "Connection: close\r\n"
            << "\r\n";

    const std::string req_str = request.str();
    ::send(fd, req_str.data(), req_str.size(), 0);

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
    ::close(fd);

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

HttpResponse httpPost(const std::string &host, int port, const std::string &path,
                      const std::string &body, int timeout_ms) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }

    struct timeval timeout;
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    struct hostent *server = ::gethostbyname(host.c_str());
    if (server == nullptr) {
        ::close(fd);
        throw std::runtime_error("host resolution failed: " + host);
    }

    struct sockaddr_in address {};
    address.sin_family = AF_INET;
    std::memcpy(&address.sin_addr.s_addr, server->h_addr, static_cast<size_t>(server->h_length));
    address.sin_port = htons(static_cast<uint16_t>(port));

    if (::connect(fd, reinterpret_cast<struct sockaddr *>(&address), sizeof(address)) < 0) {
        ::close(fd);
        throw std::runtime_error("connect failed to " + host + ":" + std::to_string(port));
    }

    std::ostringstream request;
    request << "POST " << path << " HTTP/1.1\r\n"
            << "Host: " << host << "\r\n"
            << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Connection: close\r\n"
            << "\r\n"
            << body;

    const std::string req_str = request.str();
    ::send(fd, req_str.data(), req_str.size(), 0);

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
    ::close(fd);

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
        meta.addInput(input["name"].get<std::string>(), shape, 1);
    }
    for (const auto &output : json["outputs"]) {
        std::vector<int64_t> shape;
        for (const auto &dim : output["shape"]) {
            shape.push_back(dim.get<int64_t>());
        }
        meta.addOutput(output["name"].get<std::string>(), shape, 1);
    }
    return true;
}

} // namespace

KserveClient::KserveClient(const std::string &endpoint, const std::string &model_name,
                           const std::string &model_version, int timeout_ms)
    : InferenceInterface("", false, 1), endpoint_(endpoint), model_name_(model_name),
      model_version_(model_version), timeout_ms_(timeout_ms) {}

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
KserveClient::get_infer_results(const std::vector<std::vector<uint8_t>> &input_tensors) {
    if (!metadata_loaded_) {
        throw std::runtime_error("KServe client metadata not loaded");
    }

    const auto host = extractHost(endpoint_);
    const auto port = extractPort(endpoint_, 8080);

    Json request;
    request["id"] = "kserve-client";
    Json inputs = Json::array();

    const auto &meta_inputs = cached_metadata_.getInputs();
    for (size_t i = 0; i < input_tensors.size() && i < meta_inputs.size(); ++i) {
        Json input;
        input["name"] = meta_inputs[i].name;
        input["datatype"] = "FP32";
        Json shape = Json::array();
        for (const auto dim : meta_inputs[i].shape) {
            shape.push_back(dim);
        }
        input["shape"] = shape;

        Json data = Json::array();
        const auto &bytes = input_tensors[i];
        const auto *floats = reinterpret_cast<const float *>(bytes.data());
        const size_t count = bytes.size() / sizeof(float);
        for (size_t j = 0; j < count; ++j) {
            data.push_back(floats[j]);
        }
        input["data"] = data;
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

    const std::string path =
        "/v2/models/" + model_name_ + "/versions/" + model_version_ + "/infer";
    const auto http_response =
        httpPost(host, port, path, request.dump(), timeout_ms_);

    if (http_response.status != 200) {
        throw std::runtime_error("KServe inference failed: HTTP " +
                                 std::to_string(http_response.status) + " " +
                                 http_response.body);
    }

    auto json_response = Json::parse(http_response.body);
    if (!json_response.contains("outputs") || !json_response["outputs"].is_array()) {
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

        std::vector<TensorElement> data;
        if (output.contains("data") && output["data"].is_array()) {
            for (const auto &value : output["data"]) {
                data.push_back(static_cast<float>(value.get<double>()));
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

bool KserveClient::is_gpu_available() const noexcept {
    return false;
}

bool KserveClient::fetchMetadata() {
    const auto host = extractHost(endpoint_);
    const auto port = extractPort(endpoint_, 8080);
    const std::string path = "/v2/models/" + model_name_;

    const auto http_response = httpGet(host, port, path, timeout_ms_);

    if (http_response.status != 200) {
        throw std::runtime_error("KServe metadata fetch failed: HTTP " +
                                 std::to_string(http_response.status));
    }

    metadata_loaded_ = parseMetadataJson(http_response.body, cached_metadata_);
    return metadata_loaded_;
}
