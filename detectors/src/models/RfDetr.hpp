#pragma once
#include "Detector.hpp"
class RfDetr : public Detector
{

public:
    RfDetr(const ModelInfo& model_info, float confidenceThreshold = 0.25);


    cv::Mat preprocess_image(const cv::Mat& image) override;
    std::vector<Detection> postprocess(const std::vector<std::vector<TensorElement>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) override;
    inline float sigmoid(float x) const noexcept {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    private:
        std::optional<size_t> dets_idx_; // 91 x 4
        std::optional<size_t> labels_idx_;    // 91
};