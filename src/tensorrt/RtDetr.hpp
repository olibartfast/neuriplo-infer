#pragma once
#include "TRTInfer.hpp"

class RtDetr : public TRTInfer
{


public:
    RtDetr(const std::string& model_path, bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);
 
    std::vector<float> preprocess_image(const cv::Mat& image) override; 
    std::vector<Detection> postprocess(const cv::Size& frame_size) override;  
};
 