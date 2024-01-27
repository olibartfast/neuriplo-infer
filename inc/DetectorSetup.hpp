#pragma once
#include "common.hpp"
#include "YoloV4.hpp"
#include "YoloVn.hpp"
#include "YoloNas.hpp"
#include "YoloV8.hpp"
#include "RtDetr.hpp"


std::unique_ptr<Detector> createDetector(
    const std::string& detectorType)
 {
    std::unique_ptr<Detector> detector{nullptr};
    if(detectorType.find("yolov8") != std::string::npos)  
    {
        return std::make_unique<YoloV8>();
    } 
    else if(detectorType.find("yolov4") != std::string::npos)
    {  
        return std::make_unique<YoloV4>();
    }   
    else if(detectorType.find("yolov5") != std::string::npos || 
        detectorType.find("yolov6") != std::string::npos  ||
        detectorType.find("yolov7") != std::string::npos)  
    {
        return std::make_unique<YoloVn>();
    }
    else if(detectorType.find("yolonas") != std::string::npos)  
    {
        return std::make_unique<YoloNas>();
    }    
    return detector;
}
