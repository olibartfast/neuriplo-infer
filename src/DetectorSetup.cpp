#include "DetectorSetup.hpp"


std::unique_ptr<Detector> createDetector(
    const std::string& detectorType)
 {
    std::unique_ptr<Detector> detector{nullptr};
    if(detectorType.find("yolov4") != std::string::npos)
    {  
        return std::make_unique<YoloV4>();
    }  
    else if(detectorType.find("yolov5") != std::string::npos || 
        detectorType.find("yolov6") != std::string::npos  ||
        detectorType.find("yolov7") != std::string::npos ||
        detectorType.find("yolov8") != std::string::npos ||
        detectorType.find("yolov9") != std::string::npos)  
    {
        return std::make_unique<YoloVn>();
    }
    else if(detectorType.find("yolonas") != std::string::npos)  
    {
        return std::make_unique<YoloNas>();
    } 
    else if(detectorType.find("yolov10") != std::string::npos)  
    {
        return std::make_unique<YOLOv10>();
    }     
    else if(detectorType.find("rtdetrul") != std::string::npos)  
    {
        return std::make_unique<RtDetrUltralytics>();
    }     
    else if(detectorType.find("rtdetr") != std::string::npos)  
    {
        return std::make_unique<RtDetr>();
    }     
    return detector;
}
