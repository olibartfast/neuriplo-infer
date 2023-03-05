#include "TFDetectionAPI.hpp"


TFDetectionAPI::TFDetectionAPI(
        const std::vector<std::string>& classNames,
        std::string modelConfiguration, 
        std::string modelBinary,      
        float confidenceThreshold,    
        size_t network_width,
        size_t network_height,
        float meanVal) : 
        meanVal_ {meanVal}, 
        Detector{classNames, 
        modelConfiguration, modelBinary, confidenceThreshold,
        network_width,
        network_height}
{       

}

std::vector<Detection> TFDetectionAPI::TFDetectionAPI run_detection(const cv::Mat& frame) 
{
        return std::vector<Detection>{};
}