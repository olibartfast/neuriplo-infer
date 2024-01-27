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
    ov::Shape s = compiled_model_.input().get_shape();
    channels_ = s[1];
}

std::vector<Detection> OVInfer::run_detection(const cv::Mat& image)
{
    
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;

    // Get input port for model with one input
    std::vector<float> input_data = preprocess_image(image);
    ov::Tensor input_tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), input_data.data());
    // Set input tensor for model with one input
    infer_request_.set_input_tensor(input_tensor);    
    infer_request_.infer();
    auto output_tensor = infer_request_.get_output_tensor();
    const float *output_buffer = output_tensor.data<const float>();
    std::size_t output_size = output_tensor.get_size();
    std::vector<int64_t>output_shape(output_tensor.get_shape().begin(), output_tensor.get_shape().end());
    std::vector<float> output(output_buffer, output_buffer + output_size);
    outputs.emplace_back(output);
    shapes.emplace_back(output_shape);
    return postprocess(outputs, shapes, cv::Size(image.cols, image.rows));
}