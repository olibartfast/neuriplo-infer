#pragma once
#include "Detector.hpp"
class YoloVn : public Detector{ 

public:
    YoloVn(
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);    
        
    std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) override;
    cv::Mat preprocess_image(const cv::Mat& image) override; 

    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
    {
        float r_w = network_width_ / static_cast<float>(imgSz.width);
        float r_h = network_height_ / static_cast<float>(imgSz.height);
        
        int l, r, t, b;
        if (r_h > r_w) {
            l = bbox[0] - bbox[2] / 2.f;
            r = bbox[0] + bbox[2] / 2.f;
            t = bbox[1] - bbox[3] / 2.f - (network_height_ - r_w * imgSz.height) / 2;
            b = bbox[1] + bbox[3] / 2.f - (network_height_ - r_w * imgSz.height) / 2;
            l /= r_w;
            r /= r_w;
            t /= r_w;
            b /= r_w;
        }
        else {
            l = bbox[0] - bbox[2] / 2.f - (network_width_ - r_h * imgSz.width) / 2;
            r = bbox[0] + bbox[2] / 2.f - (network_width_ - r_h * imgSz.width) / 2;
            t = bbox[1] - bbox[3] / 2.f;
            b = bbox[1] + bbox[3] / 2.f;
            l /= r_h;
            r /= r_h;
            t /= r_h;
            b /= r_h;
    }

        // Clamp the coordinates within the image bounds
        l = std::max(0, std::min(l, imgSz.width - 1));
        r = std::max(0, std::min(r, imgSz.width - 1));
        t = std::max(0, std::min(t, imgSz.height - 1));
        b = std::max(0, std::min(b, imgSz.height - 1));

        return cv::Rect(l, t, r - l, b - t);
    }
};