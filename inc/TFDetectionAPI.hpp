#include "Detector.hpp"

class TFDetectionAPI : public Detector{

public:
    std::vector<Detection> run_detection(const cv::Mat& frame) override;
};