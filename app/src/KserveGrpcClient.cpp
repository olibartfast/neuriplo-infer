#include "KserveGrpcClient.hpp"

#include "kserve_grpc.grpc.pb.h"

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "KserveProtocol.hpp"
#include "KserveRetry.hpp"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace kserve {

namespace {

std::string envToken() {
  const char *token = std::getenv("KSERVE_BEARER_TOKEN");
  return token != nullptr ? std::string(token) : std::string();
}

// gRPC sends raw byte payloads (raw_input_contents) by default — it is the
// efficient form the server already returns. Disable with a falsey
// KSERVE_BINARY (0/false/off/no) to fall back to typed `contents`.
bool rawContentsEnabled() {
  const char *raw = std::getenv("KSERVE_BINARY");
  if (raw == nullptr) {
    return true;
  }
  std::string value;
  for (const char *p = raw; *p != '\0'; ++p) {
    value.push_back(
        static_cast<char>(::tolower(static_cast<unsigned char>(*p))));
  }
  return !(value == "0" || value == "false" || value == "off" || value == "no");
}

// Applies the shared per-call deadline and optional bearer token.
void prepareContext(grpc::ClientContext &context, int timeout_ms,
                    const std::string &auth_token) {
  context.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::milliseconds(timeout_ms));
  if (!auth_token.empty()) {
    context.AddMetadata("authorization", "Bearer " + auth_token);
  }
}

[[noreturn]] void throwStatus(const char *what, const grpc::Status &status) {
  std::ostringstream msg;
  msg << what << ": " << static_cast<int>(status.error_code()) << " "
      << status.error_message();
  throw std::runtime_error(msg.str());
}

// Wraps a unary stub call in the retry/backoff loop. `call()` must build a
// fresh ClientContext, reset its response, and return the resulting
// grpc::Status; transient codes (UNAVAILABLE / DEADLINE_EXCEEDED /
// RESOURCE_EXHAUSTED) are retried, and the final status is returned so callers
// throw as before.
template <typename Call>
grpc::Status retryGrpc(const RetryPolicy &policy, Call call) {
  return runWithRetry(
      policy, call,
      [](const grpc::Status &s) {
        return !s.ok() &&
               isRetryableGrpcStatus(static_cast<int>(s.error_code()));
      },
      defaultSleep, defaultJitter);
}

// Reinterprets raw little-endian bytes as `datatype` and fills the matching
// typed field of an InferTensorContents.
template <typename T, typename Add>
void fillContents(const std::vector<uint8_t> &bytes, Add add) {
  if (bytes.size() % sizeof(T) != 0) {
    throw std::runtime_error(
        "KServe gRPC input byte count is not a multiple of "
        "the datatype width");
  }
  const size_t count = bytes.size() / sizeof(T);
  for (size_t i = 0; i < count; ++i) {
    T value;
    std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));
    add(value);
  }
}

void encodeInput(inference::InferTensorContents *contents,
                 const std::vector<uint8_t> &bytes,
                 const std::string &datatype) {
  if (datatype == "FP32") {
    fillContents<float>(bytes,
                        [&](float v) { contents->add_fp32_contents(v); });
  } else if (datatype == "FP64") {
    fillContents<double>(bytes,
                         [&](double v) { contents->add_fp64_contents(v); });
  } else if (datatype == "INT8") {
    fillContents<int8_t>(bytes,
                         [&](int8_t v) { contents->add_int_contents(v); });
  } else if (datatype == "INT16") {
    fillContents<int16_t>(bytes,
                          [&](int16_t v) { contents->add_int_contents(v); });
  } else if (datatype == "INT32") {
    fillContents<int32_t>(bytes,
                          [&](int32_t v) { contents->add_int_contents(v); });
  } else if (datatype == "INT64") {
    fillContents<int64_t>(bytes,
                          [&](int64_t v) { contents->add_int64_contents(v); });
  } else if (datatype == "UINT8" || datatype == "BOOL") {
    fillContents<uint8_t>(bytes,
                          [&](uint8_t v) { contents->add_uint_contents(v); });
  } else if (datatype == "UINT16") {
    fillContents<uint16_t>(bytes,
                           [&](uint16_t v) { contents->add_uint_contents(v); });
  } else if (datatype == "UINT32") {
    fillContents<uint32_t>(bytes,
                           [&](uint32_t v) { contents->add_uint_contents(v); });
  } else if (datatype == "UINT64") {
    fillContents<uint64_t>(
        bytes, [&](uint64_t v) { contents->add_uint64_contents(v); });
  } else {
    throw std::runtime_error(
        "unsupported KServe gRPC input datatype (use HTTP transport): " +
        datatype);
  }
}

// Appends each element of a typed contents range as raw little-endian bytes of
// width T (used when the server returns typed contents instead of raw bytes).
template <typename T, typename Range>
void appendBytes(std::vector<uint8_t> &out, const Range &range) {
  for (const auto value : range) {
    const T typed = static_cast<T>(value);
    const auto *p = reinterpret_cast<const uint8_t *>(&typed);
    out.insert(out.end(), p, p + sizeof(T));
  }
}

std::vector<uint8_t> contentsToBytes(const inference::InferTensorContents &c,
                                     const std::string &datatype) {
  std::vector<uint8_t> out;
  if (datatype == "FP32") {
    appendBytes<float>(out, c.fp32_contents());
  } else if (datatype == "FP64") {
    appendBytes<double>(out, c.fp64_contents());
  } else if (datatype == "INT8") {
    appendBytes<int8_t>(out, c.int_contents());
  } else if (datatype == "INT16") {
    appendBytes<int16_t>(out, c.int_contents());
  } else if (datatype == "INT32") {
    appendBytes<int32_t>(out, c.int_contents());
  } else if (datatype == "INT64") {
    appendBytes<int64_t>(out, c.int64_contents());
  } else if (datatype == "UINT8") {
    appendBytes<uint8_t>(out, c.uint_contents());
  } else if (datatype == "BOOL") {
    appendBytes<uint8_t>(out, c.bool_contents());
  } else if (datatype == "UINT16") {
    appendBytes<uint16_t>(out, c.uint_contents());
  } else if (datatype == "UINT32") {
    appendBytes<uint32_t>(out, c.uint_contents());
  } else if (datatype == "UINT64") {
    appendBytes<uint64_t>(out, c.uint64_contents());
  } else {
    throw std::runtime_error("unsupported KServe gRPC output datatype: " +
                             datatype);
  }
  return out;
}

template <typename Range> std::vector<int64_t> toShape(const Range &dims) {
  std::vector<int64_t> shape;
  for (const auto dim : dims) {
    shape.push_back(static_cast<int64_t>(dim));
  }
  return shape;
}

} // namespace

struct GrpcClient::Impl {
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<inference::GRPCInferenceService::Stub> stub;
  RetryPolicy retry_policy = retryPolicyFromEnv();
};

GrpcClient::GrpcClient(const std::string &endpoint, std::string model_name,
                       std::string model_version, int timeout_ms)
    : model_name_(std::move(model_name)),
      model_version_(std::move(model_version)), timeout_ms_(timeout_ms),
      auth_token_(envToken()), impl_(std::make_unique<Impl>()) {
  const auto ep = parseEndpoint(endpoint, 8001);
  const std::string target = ep.host + ":" + std::to_string(ep.port);
  const auto credentials =
      ep.tls ? grpc::SslCredentials(grpc::SslCredentialsOptions{})
             : grpc::InsecureChannelCredentials();
  impl_->channel = grpc::CreateChannel(target, credentials);
  impl_->stub = inference::GRPCInferenceService::NewStub(impl_->channel);
}

GrpcClient::~GrpcClient() = default;

ModelMetadata GrpcClient::modelMetadata() {
  inference::ModelMetadataRequest request;
  request.set_name(model_name_);
  if (!model_version_.empty()) {
    request.set_version(model_version_);
  }

  inference::ModelMetadataResponse response;
  const auto status = retryGrpc(impl_->retry_policy, [&] {
    grpc::ClientContext context;
    prepareContext(context, timeout_ms_, auth_token_);
    response.Clear();
    return impl_->stub->ModelMetadata(&context, request, &response);
  });
  if (!status.ok()) {
    throwStatus("KServe gRPC metadata fetch failed", status);
  }

  ModelMetadata metadata;
  for (const auto &input : response.inputs()) {
    metadata.inputs.push_back(
        {input.name(), input.datatype().empty() ? "FP32" : input.datatype(),
         toShape(input.shape())});
  }
  for (const auto &output : response.outputs()) {
    metadata.outputs.push_back(
        {output.name(), output.datatype().empty() ? "FP32" : output.datatype(),
         toShape(output.shape())});
  }
  return metadata;
}

std::vector<InferOutput>
GrpcClient::infer(const std::vector<InferInput> &inputs) {
  inference::ModelInferRequest request;
  request.set_model_name(model_name_);
  request.set_model_version(model_version_);
  request.set_id("kserve-grpc-client");

  // Binary tensor extension: send raw little-endian bytes via
  // raw_input_contents (one entry per input, in input order) instead of the
  // typed `contents` fields. This is the efficient default and also widens
  // datatype coverage (e.g. FP16/BF16) since no typed field is required.
  const bool raw_contents = rawContentsEnabled();
  for (const auto &input : inputs) {
    if (input.data == nullptr) {
      throw std::runtime_error("KServe input '" + input.name + "' has no data");
    }
    auto *node = request.add_inputs();
    node->set_name(input.name);
    node->set_datatype(input.datatype);
    for (const auto dim : input.shape) {
      node->add_shape(dim);
    }
    if (raw_contents) {
      request.add_raw_input_contents(
          std::string(input.data->begin(), input.data->end()));
    } else {
      encodeInput(node->mutable_contents(), *input.data, input.datatype);
    }
  }

  inference::ModelInferResponse response;
  const auto status = retryGrpc(impl_->retry_policy, [&] {
    grpc::ClientContext context;
    prepareContext(context, timeout_ms_, auth_token_);
    response.Clear();
    return impl_->stub->ModelInfer(&context, request, &response);
  });
  if (!status.ok()) {
    throwStatus("KServe gRPC inference failed", status);
  }

  // The server may return tensor payloads either as raw_output_contents (the
  // common, efficient form) or as typed contents. Handle both.
  const bool has_raw = response.raw_output_contents_size() > 0;

  std::vector<InferOutput> outputs;
  for (int i = 0; i < response.outputs_size(); ++i) {
    const auto &output = response.outputs(i);
    InferOutput out;
    out.name = output.name();
    out.datatype = output.datatype().empty() ? "FP32" : output.datatype();
    out.shape = toShape(output.shape());
    if (has_raw && i < response.raw_output_contents_size()) {
      const auto &raw = response.raw_output_contents(i);
      out.data.assign(raw.begin(), raw.end());
    } else {
      out.data = contentsToBytes(output.contents(), out.datatype);
    }
    outputs.push_back(std::move(out));
  }
  return outputs;
}

bool GrpcClient::serverLive() {
  inference::ServerLiveRequest request;
  inference::ServerLiveResponse response;
  const auto status = retryGrpc(impl_->retry_policy, [&] {
    grpc::ClientContext context;
    prepareContext(context, timeout_ms_, auth_token_);
    response.Clear();
    return impl_->stub->ServerLive(&context, request, &response);
  });
  if (!status.ok()) {
    throwStatus("KServe gRPC ServerLive failed", status);
  }
  return response.live();
}

bool GrpcClient::serverReady() {
  inference::ServerReadyRequest request;
  inference::ServerReadyResponse response;
  const auto status = retryGrpc(impl_->retry_policy, [&] {
    grpc::ClientContext context;
    prepareContext(context, timeout_ms_, auth_token_);
    response.Clear();
    return impl_->stub->ServerReady(&context, request, &response);
  });
  if (!status.ok()) {
    throwStatus("KServe gRPC ServerReady failed", status);
  }
  return response.ready();
}

bool GrpcClient::modelReady() {
  inference::ModelReadyRequest request;
  request.set_name(model_name_);
  if (!model_version_.empty()) {
    request.set_version(model_version_);
  }
  inference::ModelReadyResponse response;
  const auto status = retryGrpc(impl_->retry_policy, [&] {
    grpc::ClientContext context;
    prepareContext(context, timeout_ms_, auth_token_);
    response.Clear();
    return impl_->stub->ModelReady(&context, request, &response);
  });
  if (!status.ok()) {
    throwStatus("KServe gRPC ModelReady failed", status);
  }
  return response.ready();
}

} // namespace kserve
