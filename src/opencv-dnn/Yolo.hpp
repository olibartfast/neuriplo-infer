#pragma once
#include "Detector.hpp"

class Yolo : public Detector
{
public:
    Yolo(const std::string& model_path, bool use_gpu = false,
         float confidenceThreshold = 0.25,
         size_t network_width = 640,
         size_t network_height = 640);

    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);
    std::vector<float> preprocess_image(const cv::Mat& image);
    std::vector<Detection> postprocess(const float* output0, const std::vector<int64_t>& shape0, const cv::Size& frame_size);
    std::vector<Detection> postprocess(const float* output0, const float* output1 ,const std::vector<int64_t>& shape0, const std::vector<int64_t>& shape1, const cv::Size& frame_size);
    cv::Mat preprocess_image_mat(const cv::Mat& img);
};
