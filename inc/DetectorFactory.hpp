#pragma once
#include "common.hpp"
#ifdef USE_TENSORFLOW
#include "TFDetectionAPI.hpp"
#elif USE_OPENCV_DNN
#include "YoloV8.hpp"
#include "YoloV4.hpp"
#include "YoloVn.hpp"
#include "YoloNas.hpp"
#elif USE_ONNX_RUNTIME
#include "YoloV8.hpp"
#include "YoloNas.hpp"
#include "RtDetr.hpp"
#include "YoloVn.hpp"
#elif USE_LIBTORCH
#include "YoloV8.hpp"
#include "RtDetr.hpp"
#include "YoloVn.hpp"
#else // supported from all backends
#include "YoloV8.hpp"
#include "RtDetr.hpp"
#endif
#include "utils.hpp"

std::unique_ptr<Detector> createDetector(
    const std::string& detectorType,
    bool use_gpu,
    const std::string& labels,
    const std::string& weights,
    const std::string& modelConfiguration = "")
 {
#ifdef USE_TENSORFLOW      
    if(detectorType.find("tensorflow") != std::string::npos) 
    {
        if(isDirectory(weights))
            return std::make_unique<TFDetectionAPI>(weights);
        else
        {
            std::cerr << "In case of Tensorflow weights must be a path to the saved model folder" << std::endl;
            return nullptr;   
        }    
    }

      
#elif USE_OPENCV_DNN
    if(detectorType.find("yolov8") != std::string::npos)  
    {
        return std::make_unique<YoloV8>(weights, use_gpu);
    } 
    else if(detectorType.find("yolov4") != std::string::npos)
    {
        if(modelConfiguration.empty() || !std::filesystem::exists(modelConfiguration))
        {
            std::cerr << "YoloV4 needs a configuration file" << std::endl;
            return nullptr;
        }    
        return std::make_unique<YoloV4>(modelConfiguration, weights);
    }   
    else if(detectorType.find("yolov5") != std::string::npos || 
        detectorType.find("yolov6") != std::string::npos  ||
        detectorType.find("yolov7") != std::string::npos)  
    {
        return std::make_unique<YoloVn>(weights);
    }
    else if(detectorType.find("yolonas") != std::string::npos)  
    {
        return std::make_unique<YoloNas>(weights);
    }  
#elif USE_ONNX_RUNTIME
    if(detectorType.find("yolov8") != std::string::npos)  
    {
        return std::make_unique<YoloV8>(weights, use_gpu);
    }    
    else if(detectorType.find("yolonas") != std::string::npos)  
    {
        return std::make_unique<YoloNas>(weights, use_gpu);
    }
    else if(detectorType.find("rtdetr") != std::string::npos)  
    {
        return std::make_unique<RtDetr>(weights, use_gpu);
    }    
    else if(detectorType.find("yolov5") != std::string::npos || 
        detectorType.find("yolov6") != std::string::npos  ||
        detectorType.find("yolov7") != std::string::npos)  
    {
        return std::make_unique<YoloVn>(weights);
    }     
#elif USE_LIBTORCH
    if(detectorType.find("yolov8") != std::string::npos)  
    {
        return std::make_unique<YoloV8>(weights, use_gpu);
    }    
    else if(detectorType.find("rtdetr") != std::string::npos)  
    {
        return std::make_unique<RtDetr>(weights, use_gpu);
    } 
    else if(detectorType.find("yolov5") != std::string::npos || 
        detectorType.find("yolov6") != std::string::npos  ||
        detectorType.find("yolov7") != std::string::npos)  
    {
        return std::make_unique<YoloVn>(weights);
    }        
#else
    if(detectorType.find("yolov8") != std::string::npos)  
    {
        return std::make_unique<YoloV8>(weights, use_gpu);
    }      
    else if(detectorType.find("rtdetr") != std::string::npos)  
    {
        return std::make_unique<RtDetr>(weights, use_gpu);
    }         
#endif    
    
    else
    return nullptr;
}
