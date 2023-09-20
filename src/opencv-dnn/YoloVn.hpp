#pragma once
#include "Yolo.hpp"
class YoloVn : public Yolo{ 
protected: 
 	  cv::dnn::Net net_;   
      float score_threshold_ = 0.5;
      float nms_threshold_ = 0.4;


public:
    YoloVn(std::string modelBinary, 
		bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640); 
    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);
    std::vector<float> preprocess_image(const cv::Mat& image);       
	  std::vector<Detection> run_detection(const cv::Mat& frame) override;
};