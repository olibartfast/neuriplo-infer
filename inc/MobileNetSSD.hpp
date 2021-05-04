#include "Detector.hpp"

// MobileNet Single-Shot Detector 
// (https://arxiv.org/abs/1512.02325)
// to detect objects on image, caffemodel model's file is avaliable here:
// https://github.com/chuanqi305/MobileNet-SSD


class MobileNetSSD : public Detector
{
  cv::dnn::Net net_;
  float inScaleFactor_;
  float meanVal_; 
public:
  MobileNetSSD(
        const std::vector<std::string>& classNames,
        std::string modelConfiguration, 
        std::string modelBinary,       
        float confidenceThreshold = 0.25,    
        size_t network_width = 300,
        size_t network_height = 300,
        float inScaleFactor = 0.007843f,
        float meanVal = 127.5);

    ~MobileNetSSD(){}
    void run_detection(const cv::Mat& frame) override;

};
