#pragma once
#include "Detector.hpp"
#include <torch/torch.h>
#include <torch/script.h>

class LibtorchInfer : public Detector
{
protected:
    torch::DeviceType device_;
    torch::jit::script::Module module_;

public:
    LibtorchInfer(const std::string& model_path, bool use_gpu = true,
    float confidenceThreshold = 0.25,
    size_t network_width = 640,
    size_t network_height = 640);

    std::vector<Detection> run_detection(const cv::Mat& image) override;
    virtual std::vector<float> preprocess_image(const cv::Mat& image) = 0; 
    virtual std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) = 0;  
  
};