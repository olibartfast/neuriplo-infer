#pragma once
#include "Yolo.hpp"
#include <torch/torch.h>
#include <torch/script.h>

class YoloV8 : public Yolo
{
private:
    torch::DeviceType device_;
    torch::jit::script::Module module_;

public:

    YoloV8(const std::string& model_path, bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640) : 
        Yolo{model_path, use_gpu, confidenceThreshold,
                network_width,
                network_height}
    {
        logger_->info("Initializing YoloV8 Libtorch");
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

        // In libtorch i didn't find a way to get the input layer
        input_width_ = 640;
        input_height_= 640;
        channels_=3;

    }

    std::vector<Detection>  run_detection(const cv::Mat& image) override
    {
        // Preprocess the input image
        std::vector<float> input_tensor = preprocess_image(image);

        // Convert the input tensor to a Torch tensor
        torch::Tensor input = torch::from_blob(input_tensor.data(), { 1, channels_, input_height_, input_width_ }, torch::kFloat32);
        input = input.to(device_);

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto output = module_.forward(inputs);
        torch::Tensor output_tensor0 = output.toTensor();
 
        // Convert the output tensors to CPU and extract data
        output_tensor0 = output_tensor0.to(torch::kCPU).contiguous();
        const float* output0 = output_tensor0.data_ptr<float>();
    
        // Get the shapes of the output tensors
        std::vector<int64_t> shape0 = output_tensor0.sizes().vec();
        cv::Size frame_size(image.cols, image.rows);
        return postprocess(output0, shape0, frame_size);

    }
};