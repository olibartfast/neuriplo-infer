#pragma once
#include "Detector.hpp"
#include <openvino/openvino.hpp>


class OVInfer 
{
protected:


public:
    OVInfer(const std::string& model_path, bool use_gpu = true,
    float confidenceThreshold = 0.25,
    size_t network_width = 640,
    size_t network_height = 640);

    std::vector<Detection> run_detection(const cv::Mat& image) override;
  
    ov::Core core_;
    ov::Tensor input_tensor_;
    ov::InferRequest infer_request_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
};