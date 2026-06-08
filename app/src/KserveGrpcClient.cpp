#include "KserveGrpcClient.hpp"

#include "kserve_grpc.grpc.pb.h"

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "KserveProtocol.hpp"

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
  return readSecretFromEnvOrFile("KSERVE_BEARER_TOKEN",
                                 "KSERVE_BEARER_TOKEN_FILE");
}

// Builds channel credentials for the endpoint. Plaintext for grpc://; for
// grpcs:// a TLS channel verifying the server against KSERVE_CA_CERT (or the
// system roots when unset). When KSERVE_CLIENT_CERT and KSERVE_CLIENT_KEY are
// both provided the channel presents a client certificate for mTLS.
std::shared_ptr<grpc::ChannelCredentials> makeChannelCredentials(bool tls) {
  if (!tls) {
    return grpc::InsecureChannelCredentials();
  }
  grpc::SslCredentialsOptions options;
  options.pem_root_certs = readFileFromEnvPath("KSERVE_CA_CERT");
  options.pem_cert_chain = readFileFromEnvPath("KSERVE_CLIENT_CERT");
  options.pem_private_key = readFileFromEnvPath("KSERVE_CLIENT_KEY");
  // mTLS requires both the client cert and key (or neither).
  requireClientCertPair(options.pem_cert_chain, options.pem_private_key);
  return grpc::SslCredentials(options);
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
};

GrpcClient::GrpcClient(const std::string &endpoint, std::string model_name,
                       std::string model_version, int timeout_ms)
    : model_name_(std::move(model_name)),
      model_version_(std::move(model_version)), timeout_ms_(timeout_ms),
      auth_token_(envToken()), impl_(std::make_unique<Impl>()) {
  const auto ep = parseEndpoint(endpoint, 8001);
  const std::string target = ep.host + ":" + std::to_string(ep.port);
  impl_->channel = grpc::CreateChannel(target, makeChannelCredentials(ep.tls));
  impl_->stub = inference::GRPCInferenceService::NewStub(impl_->channel);
}

GrpcClient::~GrpcClient() = default;

ModelMetadata GrpcClient::modelMetadata() {
  inference::ModelMetadataRequest request;
  request.set_name(model_name_);
  if (!model_version_.empty()) {
    request.set_version(model_version_);
  }

  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::milliseconds(timeout_ms_));
  if (!auth_token_.empty()) {
    context.AddMetadata("authorization", "Bearer " + auth_token_);
  }

  inference::ModelMetadataResponse response;
  const auto status = impl_->stub->ModelMetadata(&context, request, &response);
  if (!status.ok()) {
    std::ostringstream msg;
    msg << "KServe gRPC metadata fetch failed: "
        << static_cast<int>(status.error_code()) << " "
        << status.error_message();
    throw std::runtime_error(msg.str());
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
    encodeInput(node->mutable_contents(), *input.data, input.datatype);
  }

  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::milliseconds(timeout_ms_));
  if (!auth_token_.empty()) {
    context.AddMetadata("authorization", "Bearer " + auth_token_);
  }

  inference::ModelInferResponse response;
  const auto status = impl_->stub->ModelInfer(&context, request, &response);
  if (!status.ok()) {
    std::ostringstream msg;
    msg << "KServe gRPC inference failed: "
        << static_cast<int>(status.error_code()) << " "
        << status.error_message();
    throw std::runtime_error(msg.str());
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

namespace {

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

} // namespace

bool GrpcClient::serverLive() {
  inference::ServerLiveRequest request;
  grpc::ClientContext context;
  prepareContext(context, timeout_ms_, auth_token_);
  inference::ServerLiveResponse response;
  const auto status = impl_->stub->ServerLive(&context, request, &response);
  if (!status.ok()) {
    throwStatus("KServe gRPC ServerLive failed", status);
  }
  return response.live();
}

bool GrpcClient::serverReady() {
  inference::ServerReadyRequest request;
  grpc::ClientContext context;
  prepareContext(context, timeout_ms_, auth_token_);
  inference::ServerReadyResponse response;
  const auto status = impl_->stub->ServerReady(&context, request, &response);
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
  grpc::ClientContext context;
  prepareContext(context, timeout_ms_, auth_token_);
  inference::ModelReadyResponse response;
  const auto status = impl_->stub->ModelReady(&context, request, &response);
  if (!status.ok()) {
    throwStatus("KServe gRPC ModelReady failed", status);
  }
  return response.ready();
}

} // namespace kserve
