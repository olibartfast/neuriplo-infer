#pragma once
#include "Detector.hpp"
class DFine : public Detector
{
public:
    DFine(const ModelInfo& model_info, float confidenceThreshold = 0.25);
    cv::Mat preprocess_image(const cv::Mat& image) override;
    std::vector<Detection> postprocess(const std::vector<std::vector<TensorElement>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) override;    
};