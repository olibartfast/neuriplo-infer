#pragma once
#include "Detector.hpp"

class OCVDNNInfer : public Detector
{
protected:
	cv::dnn::Net net_;
    std::vector<int> outLayers_;
    std::string outLayerType_;
    std::vector<std::string> outNames_;
        
public:
    OCVDNNInfer(const std::string& modelConfiguration, 
         const std::string& modelBinary,
         bool use_gpu = false,
         float confidenceThreshold = 0.25,
         size_t network_width = 640,
         size_t network_height = 640);

    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);
    std::vector<Detection> run_detection(const cv::Mat& frame) override;
    virtual std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) = 0;
    virtual cv::Mat preprocess_image(const cv::Mat& image) = 0; 
};
