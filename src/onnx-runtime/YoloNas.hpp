#pragma once
#include "Yolo.hpp"
#include <onnxruntime_cxx_api.h>  // for ONNX Runtime C++ API
#include <onnxruntime_c_api.h>    // for CUDA execution provider (if using CUDA)

class YoloNas : public Yolo
{
private:
    Ort::Env env_;
    Ort::Session session_{ nullptr };
    std::vector<std::string> input_names_;  // Store input layer names
    std::vector<std::string> output_names_; // Store output layer names
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

public:
    // pretty prints a shape dimension vector
    std::string print_shape(const std::vector<std::int64_t>& v);
    YoloNas(const std::string& model_path, bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);

    std::vector<Detection> run_detection(const cv::Mat& image) override;

    std::vector<Detection> postprocess(const float* output0, const float* output1 ,const std::vector<int64_t>& shape0, const std::vector<int64_t>& shape1, const cv::Size& frame_size);
    std::vector<float> preprocess_image(const cv::Mat& image);
};