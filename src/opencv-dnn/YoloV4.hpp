#include "Detector.hpp"

class YoloV4 : public Detector{
	  cv::dnn::Net net_;

public:
    YoloV4(std::string modelConfiguration, 
        std::string modelBinary, 
        bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 608,
        size_t network_height = 608); 
	std::vector<Detection> run_detection(const cv::Mat& frame) override;
};