#include "KserveGrpcClient.hpp"

#include "kserve_grpc.grpc.pb.h"

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace grpc_client {

struct KserveGrpcClient::Impl {
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<inference::GRPCInferenceService::Stub> stub;
};

KserveGrpcClient::KserveGrpcClient(const std::string &endpoint, const std::string &model_name,
                                   const std::string &model_version, int timeout_ms)
    : InferenceInterface("", false, 1), endpoint_(endpoint), model_name_(model_name),
      model_version_(model_version), timeout_ms_(timeout_ms),
      impl_(std::make_unique<Impl>()) {
    impl_->channel = grpc::CreateChannel(endpoint_, grpc::InsecureChannelCredentials());
    impl_->stub = inference::GRPCInferenceService::NewStub(impl_->channel);
}

KserveGrpcClient::~KserveGrpcClient() = default;

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
KserveGrpcClient::get_infer_results(const std::vector<std::vector<uint8_t>> &input_tensors) {
    if (!metadata_loaded_) {
        throw std::runtime_error("KServe gRPC client metadata not loaded");
    }

    inference::ModelInferRequest request;
    request.set_model_name(model_name_);
    request.set_model_version(model_version_);
    request.set_id("kserve-grpc-client");

    const auto &meta_inputs = cached_metadata_.getInputs();
    for (size_t i = 0; i < input_tensors.size() && i < meta_inputs.size(); ++i) {
        auto *input = request.add_inputs();
        input->set_name(meta_inputs[i].name);
        input->set_datatype("FP32");
        for (const auto dim : meta_inputs[i].shape) {
            input->add_shape(dim);
        }

        auto *contents = input->mutable_contents();
        const auto &bytes = input_tensors[i];
        const auto *floats = reinterpret_cast<const float *>(bytes.data());
        const size_t count = bytes.size() / sizeof(float);
        for (size_t j = 0; j < count; ++j) {
            contents->add_fp32_contents(floats[j]);
        }
    }

    const auto &meta_outputs = cached_metadata_.getOutputs();
    for (const auto &output : meta_outputs) {
        auto *out = request.add_outputs();
        out->set_name(output.name);
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(timeout_ms_));

    inference::ModelInferResponse response;
    const auto status = impl_->stub->ModelInfer(&context, request, &response);

    if (!status.ok()) {
        std::ostringstream msg;
        msg << "KServe gRPC inference failed: " << static_cast<int>(status.error_code()) << " "
            << status.error_message();
        throw std::runtime_error(msg.str());
    }

    std::vector<std::vector<TensorElement>> output_data;
    std::vector<std::vector<int64_t>> output_shapes;

    for (const auto &output : response.outputs()) {
        std::vector<int64_t> shape;
        for (const auto dim : output.shape()) {
            shape.push_back(dim);
        }
        output_shapes.push_back(shape);

        std::vector<TensorElement> data;
        const auto &contents = output.contents();
        for (const auto value : contents.fp32_contents()) {
            data.push_back(value);
        }
        for (const auto value : contents.fp64_contents()) {
            data.push_back(static_cast<float>(value));
        }
        for (const auto value : contents.int64_contents()) {
            data.push_back(static_cast<int64_t>(value));
        }
        for (const auto value : contents.int_contents()) {
            data.push_back(static_cast<int32_t>(value));
        }
        output_data.push_back(std::move(data));
    }

    return {std::move(output_data), std::move(output_shapes)};
}

InferenceMetadata KserveGrpcClient::get_inference_metadata() {
    if (!metadata_loaded_) {
        fetchMetadata();
    }
    return cached_metadata_;
}

bool KserveGrpcClient::is_gpu_available() const noexcept {
    return false;
}

bool KserveGrpcClient::fetchMetadata() {
    inference::ModelMetadataRequest request;
    request.set_name(model_name_);
    if (!model_version_.empty()) {
        request.set_version(model_version_);
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(timeout_ms_));

    inference::ModelMetadataResponse response;
    const auto status = impl_->stub->ModelMetadata(&context, request, &response);

    if (!status.ok()) {
        std::ostringstream msg;
        msg << "KServe gRPC metadata fetch failed: " << static_cast<int>(status.error_code())
            << " " << status.error_message();
        throw std::runtime_error(msg.str());
    }

    for (const auto &input : response.inputs()) {
        std::vector<int64_t> shape;
        for (const auto dim : input.shape()) {
            shape.push_back(dim);
        }
        cached_metadata_.addInput(input.name(), shape, 1);
    }
    for (const auto &output : response.outputs()) {
        std::vector<int64_t> shape;
        for (const auto dim : output.shape()) {
            shape.push_back(dim);
        }
        cached_metadata_.addOutput(output.name(), shape, 1);
    }

    metadata_loaded_ = true;
    return true;
}

} // namespace grpc_client
