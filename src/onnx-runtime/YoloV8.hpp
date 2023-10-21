#pragma once
#include "ORTInfer.hpp"

class YoloV8 : public ORTInfer
{
public:
    YoloV8(const std::string& model_path, bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);
    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);        
    std::vector<float> preprocess_image(const cv::Mat& image);
    std::vector<Detection> postprocess(const float* output0, const std::vector<int64_t>& shape0, const cv::Size& frame_size);
    std::vector<Detection> run_detection(const cv::Mat& image) override;
};