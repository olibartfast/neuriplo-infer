#include "OVInfer.hpp" 

OVInfer::OVInfer(const std::string& model_path, bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{model_path, use_gpu, confidenceThreshold,
            network_width,
            network_height}
{

    model_ = core_.read_model(model_path);
    compiled_model_ = core_.compile_model(model_);
    infer_request_ = compiled_model_.create_infer_request();
}