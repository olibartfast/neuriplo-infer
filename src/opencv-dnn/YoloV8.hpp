#pragma once
#include "Yolo.hpp"

class YoloV8 : public Yolo{
	  cv::dnn::Net net_;
      float score_threshold_ = 0.5;
      float nms_threshold_ = 0.4;


public:
    YoloV8(std::string modelBinary, 
        bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640); 
	std::vector<Detection> run_detection(const cv::Mat& frame) override;
};