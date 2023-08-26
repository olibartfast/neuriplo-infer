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
        size_t network_height = 640);

    std::vector<Detection> run_detection(const cv::Mat& image) override;
};