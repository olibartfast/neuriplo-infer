#include "LibtorchInfer.hpp"


LibtorchInfer::LibtorchInfer(const std::string& model_path, bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{model_path, use_gpu, confidenceThreshold,
            network_width,
            network_height}
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

    channels_ = 3;
}

std::vector<Detection> LibtorchInfer::run_detection(const cv::Mat& image)
{

    // Preprocess the input image
    std::vector<float> input_tensor = preprocess_image(image);

    // Convert the input tensor to a Torch tensor
    torch::Tensor input = torch::from_blob(input_tensor.data(), { 1, channels_, network_height_, network_width_ }, torch::kFloat32);
    input = input.to(device_);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto output = module_.forward(inputs);

    std::vector<std::vector<float>> output_vectors;
    std::vector<std::vector<int64_t>> shape_vectors;

    if (output.isTuple()) {
        // Handle the case where the model returns a tuple
        auto tuple_outputs = output.toTuple()->elements();

        for (const auto& output_tensor : tuple_outputs) {
            if(!output_tensor.isTensor())
                continue;
            torch::Tensor tensor = output_tensor.toTensor().to(torch::kCPU).contiguous();

            // Get the output data as a float pointer
            const float* output_data = tensor.data_ptr<float>();

            // Store the output data in the outputs vector
            std::vector<float> output_vector(output_data, output_data + tensor.numel());
            output_vectors.push_back(output_vector);

            // Get the shape of the output tensor
            std::vector<int64_t> shape = tensor.sizes().vec();
            shape_vectors.push_back(shape);
        }
    } else {
        torch::Tensor tensor = output.toTensor();
        if (tensor.size(0) == 1) {
            // If there's only one output tensor
            torch::Tensor output_tensor = tensor.to(torch::kCPU).contiguous();
            
            // Get the output data as a float pointer
            const float* output_data = output_tensor.data_ptr<float>();

            // Store the output data and shape in vectors
            output_vectors.emplace_back(output_data, output_data + output_tensor.numel());
            shape_vectors.push_back(output_tensor.sizes().vec());
        } else {
            for (int i = 0; i < tensor.size(0); ++i) {
                torch::Tensor output_tensor = tensor[i].to(torch::kCPU).contiguous();

                // Get the output data as a float pointer
                const float* output_data = output_tensor.data_ptr<float>();

                // Store the output data and shape in vectors
                output_vectors.emplace_back(output_data, output_data + output_tensor.numel());
                shape_vectors.push_back(output_tensor.sizes().vec());
            }
        }

    }

    cv::Size frame_size(image.cols, image.rows);

    // Perform post-processing on the output and return the detections
    return postprocess(output_vectors, shape_vectors, frame_size);
}
