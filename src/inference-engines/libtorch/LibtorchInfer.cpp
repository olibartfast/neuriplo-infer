#include "LibtorchInfer.hpp"


LibtorchInfer::LibtorchInfer(const std::string& model_path, bool use_gpu) : InferenceInterface{model_path, "", use_gpu}
{
    if (use_gpu && torch::cuda::is_available())
    {
        device_ = torch::kCUDA;
        logger_->info("Using CUDA GPU");
    }
    else
    {
        device_ = torch::kCPU;
        logger_->info("Using CPU");
    }

    module_ = torch::jit::load(model_path, device_);

}

std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> LibtorchInfer::get_infer_results(const cv::Mat& input_blob)
{

    // Convert the input tensor to a Torch tensor
    torch::Tensor input = torch::from_blob(input_blob.data, { 1, input_blob.size[1], input_blob.size[2], input_blob.size[3] }, torch::kFloat32);
    input = input.to(device_);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto output = module_.forward(inputs);

    std::vector<std::vector<std::any>> output_vectors;
    std::vector<std::vector<int64_t>> shape_vectors;

    if (output.isTuple()) {
        // Handle the case where the model returns a tuple
        auto tuple_outputs = output.toTuple()->elements();

        for (const auto& output_tensor : tuple_outputs) {
            if(!output_tensor.isTensor())
                continue;
            torch::Tensor tensor = output_tensor.toTensor().to(torch::kCPU).contiguous();

            // Get the output data type
            torch::ScalarType data_type = tensor.scalar_type();

            // Store the output data based on its type
            std::vector<std::any> tensor_data;
            if (data_type == torch::kFloat32) {
                const float* output_data = tensor.data_ptr<float>();
                tensor_data.reserve(tensor.numel());
                for (size_t i = 0; i < tensor.numel(); ++i) {
                    tensor_data.emplace_back(output_data[i]);
                }
            } else if (data_type == torch::kInt64) {
                const int64_t* output_data = tensor.data_ptr<int64_t>();
                tensor_data.reserve(tensor.numel());
                for (size_t i = 0; i < tensor.numel(); ++i) {
                    tensor_data.emplace_back(output_data[i]);
                }
            } else {
                // Handle other data types if needed
                std::exit(1);
            }

            // Store the output data in the outputs vector
            output_vectors.push_back(tensor_data);

            // Get the shape of the output tensor
            std::vector<int64_t> shape = tensor.sizes().vec();
            shape_vectors.push_back(shape);
        }
    } else {
        torch::Tensor tensor = output.toTensor().to(torch::kCPU).contiguous();

        // Get the output data type
        torch::ScalarType data_type = tensor.scalar_type();

        // Store the output data based on its type
        std::vector<std::any> tensor_data;
        if (data_type == torch::kFloat32) {
            const float* output_data = tensor.data_ptr<float>();
            tensor_data.reserve(tensor.numel());
            for (size_t i = 0; i < tensor.numel(); ++i) {
                tensor_data.emplace_back(output_data[i]);
            }
        } else if (data_type == torch::kInt64) {
            const int64_t* output_data = tensor.data_ptr<int64_t>();
            tensor_data.reserve(tensor.numel());
            for (size_t i = 0; i < tensor.numel(); ++i) {
                tensor_data.emplace_back(output_data[i]);
            }
        } else {
            // Handle other data types if needed
            std::exit(1);
        }

        // Store the output data in the outputs vector
        output_vectors.push_back(tensor_data);

        // Get the shape of the output tensor
        std::vector<int64_t> shape = tensor.sizes().vec();
        shape_vectors.push_back(shape);
    }

    return std::make_tuple(output_vectors, shape_vectors);
}
