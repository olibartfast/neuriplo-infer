#pragma once
#include "OVInfer.hpp"

class YoloVn : public OVInfer
{
    public:
        YoloVn(const std::string& model_path, bool use_gpu = false,
            float confidenceThreshold = 0.25,
            size_t network_width = 640,
            size_t network_height = 640);

        cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);

        std::vector<float> preprocess_image(const cv::Mat& image) override{return   std::vector<float> {};}
        std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) override
        {
            return std::vector<Detection>{};
        }    
    };
