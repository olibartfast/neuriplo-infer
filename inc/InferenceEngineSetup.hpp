#pragma once
#include "common.hpp"
#include "InferenceEngine.hpp"
#ifdef ONNX_RUNTIME
#include "ORTInfer.hpp"
#elif LIBTORCH 
#include "LibtorchInfer.hpp"
#elif LIBTENSORFLOW 
#include "TFDetectionAPI.hpp"
#elif OPENCV_DNN 
#include "OCVDNNInfer.hpp"
#elif TENSORRT
#include "TRTInfer.hpp"
#elif OPENVINO
#include "OVInfer.hpp"
#endif

std::unique_ptr<InferenceEngine> setup_inference_engine()
{

    return std::unique_ptr<OCVDNNInfer>{}; 
}