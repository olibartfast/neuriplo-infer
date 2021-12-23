#include "Detector.hpp"

class Yolo : public Detector{
	  cv::dnn::Net net_;

public:
    Yolo(const std::vector<std::string>& classNames,
 	    std::string modelConfiguration, 
        std::string modelBinary, 
        float confidenceThreshold = 0.25,
        size_t network_width = 608,
        size_t network_height = 608); 
	void run_detection(cv::Mat& frame) override;
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
};