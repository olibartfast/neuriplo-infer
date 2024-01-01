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


std::vector<Detection> YoloVn::postprocess(const std::vector<std::vector<float>>& outputs, 
    const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) 
{

    return std::vector<Detection>{};
} 

std::vector<float> YoloVn::preprocess_image(const cv::Mat &image)
{
    cv::Mat blob;
    cv::cvtColor(image, blob, cv::COLOR_BGR2RGB);
    int target_width, target_height, offset_x, offset_y;
    float resize_ratio_width = static_cast<float>(network_width_) / static_cast<float>(image.cols);
    float resize_ratio_height = static_cast<float>(network_height_) / static_cast<float>(image.rows);

    if (resize_ratio_height > resize_ratio_width)
    {
        target_width = network_width_;
        target_height = resize_ratio_width * image.rows;
        offset_x = 0;
        offset_y = (network_height_ - target_height) / 2;
    }
    else
    {
        target_width = resize_ratio_height * image.cols;
        target_height = network_height_;
        offset_x = (network_width_ - target_width) / 2;
        offset_y = 0;
    }

    cv::Mat resized_image(target_height, target_width, CV_8UC3);
    cv::resize(blob, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat output_image(network_width_, network_height_, CV_8UC3, cv::Scalar(128, 128, 128));
    resized_image.copyTo(output_image(cv::Rect(offset_x, offset_y, resized_image.cols, resized_image.rows)));   
    output_image.convertTo(output_image, CV_32FC3, 1.f / 255.f);        

    size_t img_byte_size = output_image.total() * output_image.elemSize();  // Allocate a buffer to hold all image elements.
    std::vector<float> input_data = std::vector<float>(network_width_ * network_height_ * channels_);
    std::memcpy(input_data.data(), output_image.data, img_byte_size);

    std::vector<cv::Mat> chw;
    for (size_t i = 0; i < channels_; ++i)
    {
        chw.emplace_back(cv::Mat(cv::Size(network_width_, network_height_), CV_32FC1, &(input_data[i * network_width_ * network_height_])));
    }
    cv::split(output_image, chw);

    return input_data;
}



cv::Rect YoloVn::get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
{
    float r_w = network_width_ / static_cast<float>(imgSz.width);
    float r_h = network_height_ / static_cast<float>(imgSz.height);
    
    int l, r, t, b;
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (network_height_ - r_w * imgSz.height) / 2;
        b = bbox[1] + bbox[3] / 2.f - (network_height_ - r_w * imgSz.height) / 2;
        l /= r_w;
        r /= r_w;
        t /= r_w;
        b /= r_w;
    }
    else {
        l = bbox[0] - bbox[2] / 2.f - (network_width_ - r_h * imgSz.width) / 2;
        r = bbox[0] + bbox[2] / 2.f - (network_width_ - r_h * imgSz.width) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l /= r_h;
        r /= r_h;
        t /= r_h;
        b /= r_h;
}

    // Clamp the coordinates within the image bounds
    l = std::max(0, std::min(l, imgSz.width - 1));
    r = std::max(0, std::min(r, imgSz.width - 1));
    t = std::max(0, std::min(t, imgSz.height - 1));
    b = std::max(0, std::min(b, imgSz.height - 1));

    return cv::Rect(l, t, r - l, b - t);
}
