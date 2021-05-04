#include "Detector.hpp"

class Yolo : public Detector{
	  cv::dnn::Net net_;

public:
    Yolo(const std::vector<std::string>& classNames,
 	    std::string modelConfiguration, 
        std::string modelBinary, 
        float confidenceThreshold = 0.25,
        size_t network_width = 416,
        size_t network_height = 416); 
	void run_detection(const cv::Mat& frame) override;
};