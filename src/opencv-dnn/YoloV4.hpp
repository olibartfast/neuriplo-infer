#pragma once
#include "OCVDNNInfer.hpp"

class YoloV4 : public  OCVDNNInfer{

public:
    YoloV4(std::string modelConfiguration, 
        std::string modelBinary, 
        bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 608,
        size_t network_height = 608); 
    std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) override;
    cv::Mat preprocess_image(const cv::Mat& image) override; 
};