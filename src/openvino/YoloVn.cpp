#include "YoloVn.hpp"

YoloVn::YoloVn(const std::string& model_path, bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    OVInfer{model_path, use_gpu, confidenceThreshold,
            network_width,
            network_height}
{
    logger_->info("Running openvino runtime for {}", model_path);
}
