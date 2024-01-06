#pragma once
#include "Detector.hpp"



class OCVDNNInfer : public Detector
{
protected:
    cv::dnn::Net net_;

public:
    OCVDNNInfer(const std::string& model_path, bool use_gpu = true,
    float confidenceThreshold = 0.25,
    size_t network_width = 640,
    size_t network_height = 640);

    std::vector<Detection> run_detection(const cv::Mat& image) override {
        
      
        return  std::vector<Detection> {};
    }
    virtual std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, 
        const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) = 0;  
  

};