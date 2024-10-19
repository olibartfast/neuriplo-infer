#pragma once
#include "Detector.hpp"
class YoloVn : public Detector{ 

public:
    YoloVn(
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);    
        
    std::vector<Detection> postprocess(const std::vector<std::vector<TensorElement>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) override;
    cv::Mat preprocess_image(const cv::Mat& image) override; 

    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);


    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> postprocess_v567(const TensorElement* output, const std::vector<int64_t>& shape, const cv::Size& frame_size);
    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> postprocess_ultralytics(const TensorElement* output, const std::vector<int64_t>& shape, const cv::Size& frame_size);
};