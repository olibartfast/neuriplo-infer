#pragma once
#include "Detector.hpp"

class YoloV4 : public  Detector{

public:
    YoloV4(
        float confidenceThreshold = 0.25,
        size_t network_width = 416,
        size_t network_height = 416); 
    std::vector<Detection> postprocess(const std::vector<std::vector<TensorElement>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) override;
    cv::Mat preprocess_image(const cv::Mat& image) override; 
};