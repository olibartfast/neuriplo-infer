#pragma once
#include "Detector.hpp"
#include <openvino/openvino.hpp>


class OVInfer : public Detector
{
protected:


public:
    OVInfer(const std::string& model_path, bool use_gpu = true,
    float confidenceThreshold = 0.25,
    size_t network_width = 640,
    size_t network_height = 640);

    std::vector<Detection> run_detection(const cv::Mat& image) override {
        
        // Get input port for model with one input
        std::vector<float> input_data = preprocess_image(image);
        auto input_port = compiled_model_.input();
        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), input_data.data());
        // Set input tensor for model with one input
        infer_request_.set_input_tensor(input_tensor);    
        // Get output tensor by tensor name
        auto output = infer_request_.get_tensor("output0");
        const float *output_buffer = output.data<const float>();
        // output_buffer[] - accessing output tensor data 
        return  std::vector<Detection> {};
    }
    virtual std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, 
        const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) = 0;  
  
    ov::Core core_;
    ov::Tensor input_tensor_;
    ov::InferRequest infer_request_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;


};