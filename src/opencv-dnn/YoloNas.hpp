#pragma once
#include "OCVDNNInfer.hpp"

class YoloNas : public OCVDNNInfer{

public:
    YoloNas(std::string modelBinary, 
		bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);    
        
    std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) override;
    cv::Mat preprocess_image(const cv::Mat& image) override; 
};   