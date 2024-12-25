#pragma once
#include <variant>
#include <cstdint>
#include <glog/logging.h>
#include "Detection.hpp"
#include "ModelInfo.hpp"

using TensorElement = std::variant<float, int32_t, int64_t>;

class Detector {
protected:    
    float confidenceThreshold_; 
    float nms_threshold_ = 0.4f;      
    size_t network_width_;
    size_t network_height_;    
    std::string backend_;
    int channels_{ -1 };
    ModelInfo model_info_;

public:
    Detector(ModelInfo model_info, float confidenceThreshold = 0.25)
        : model_info_{std::move(model_info)}, confidenceThreshold_{confidenceThreshold}
    {
        const auto& inputs = model_info_.getInputs();
        if (!inputs.empty()) {
            const auto& first_input = inputs[0].shape;
            if (first_input.size() >= 4) {
                channels_ = static_cast<int>(first_input[1]);
                network_height_ = static_cast<size_t>(first_input[2]);
                network_width_ = static_cast<size_t>(first_input[3]);
            } else {
                LOG(ERROR) << "Input shape does not match expected format (NCHW)";
            }
        } else {
            LOG(ERROR) << "No input layers found in model";
        }
    }

    float getConfidenceThreshold() const { return confidenceThreshold_; }
    size_t getNetworkWidth() const { return network_width_; }
    size_t getNetworkHeight() const { return network_height_; }

    virtual std::vector<Detection> postprocess(
        const std::vector<std::vector<TensorElement>>& outputs, 
        const std::vector<std::vector<int64_t>>& shapes, 
        const cv::Size& frame_size) = 0;
        
    virtual cv::Mat preprocess_image(const cv::Mat& image) = 0;
};