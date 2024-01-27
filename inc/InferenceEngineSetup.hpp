#pragma once
#include "common.hpp"
#include "InferenceEngine.hpp"
#ifdef USE_ONNX_RUNTIME
#include "ORTInfer.hpp"
#elif USE_LIBTORCH 
#include "LibtorchInfer.hpp"
#elif USE_LIBTENSORFLOW 
#include "TFDetectionAPI.hpp"
#elif USE_OPENCV_DNN 
#include "OCVDNNInfer.hpp"
#elif USE_TENSORRT
#include "TRTInfer.hpp"
#elif USE_OPENVINO
#include "OVInfer.hpp"
#endif

std::unique_ptr<InferenceEngine> setup_inference_engine(const std::string& weights, const std::string& modelConfiguration )
{
    #ifdef USE_ONNX_RUNTIME
    return std::make_unique<ORTInfer>(weights, false); 
    #elif USE_LIBTORCH 

    #elif USE_LIBTENSORFLOW 

    #elif USE_OPENCV_DNN 
    return std::make_unique<OCVDNNInfer>(weights, modelConfiguration); 
    #elif USE_TENSORRT

    #elif USE_OPENVINO

    #endif
    return nullptr;


}