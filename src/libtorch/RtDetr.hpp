#pragma once
#include "Detector.hpp"
#include <torch/torch.h>
#include <torch/script.h>

class RtDetr : public Detector
{
private:
    torch::DeviceType device_;
    torch::jit::script::Module module_;

public:
    RtDetr(const std::string& model_path, bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);

    std::vector<Detection> run_detection(const cv::Mat& image) override;
    std::vector<float> preprocess_image(const cv::Mat& image);
    std::vector<Detection> postprocess(const float* output0, const std::vector<int64_t>& shape0, const cv::Size& frame_size);    
};