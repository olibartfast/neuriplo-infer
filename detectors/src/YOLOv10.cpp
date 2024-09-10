
#include "YOLOv10.hpp"

YOLOv10::YOLOv10(
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{confidenceThreshold,
            network_width,
            network_height}
{

}


std::vector<Detection> YOLOv10::postprocess(const std::vector<std::vector<std::any>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) 
{
    const std::any* output0 = outputs.front().data();
    const  std::vector<int64_t> shape0 = shapes.front();

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    int rows = shape0[1]; // 300

    std::vector<Detection> detections;
    for (int i = 0; i < rows; ++i) 
    {

        float score = std::any_cast<float>(*(output0 + 4 ));
        if (score >= confidenceThreshold_) 
        {
            Detection det;
            float label = std::any_cast<float>(*(output0 + 5 ));
            det.label = static_cast<int>(label);
            det.score = score;
            float r_w = (frame_size.width * 1.0f) / network_width_;
            float r_h = (frame_size.height * 1.0f) / network_height_ ;
            float scale_factor = std::max(r_w, r_h);

            float pad_x = (network_width_ - frame_size.width / scale_factor) * 0.5f;
            float pad_y = (network_height_ - frame_size.height / scale_factor) * 0.5f;
            float x0 = std::any_cast<float>(*output0);
            float x1 = std::any_cast<float>(*(output0 + 1));
            float x2 = std::any_cast<float>(*(output0 + 2));
            float x3 = std::any_cast<float>(*(output0 + 3));

            float l = (x0 - pad_x) * scale_factor;
            float t = (x1 - pad_y) * scale_factor;
            float w = (x2 - x0) * scale_factor;
            float h = (x3 - x1) * scale_factor;

            int frame_width = frame_size.width;
            int frame_height = frame_size.height;

            // Clip the bounding box to ensure it stays within the image boundaries
            float l_clipped = std::max(0.0f, std::min(l, static_cast<float>(frame_width - 1)));
            float t_clipped = std::max(0.0f, std::min(t, static_cast<float>(frame_height - 1)));
            float r_clipped = std::max(0.0f, std::min(l + w, static_cast<float>(frame_width - 1)));
            float b_clipped = std::max(0.0f, std::min(t + h, static_cast<float>(frame_height - 1)));

            // Recalculate width and height after clipping
            float w_clipped = r_clipped - l_clipped;
            float h_clipped = b_clipped - t_clipped;

            // Assign the clipped rectangle to the detection
            det.bbox = cv::Rect(l_clipped, t_clipped, w_clipped, h_clipped);
            detections.emplace_back(det);
        }
        output0 += shape0[2];
    }
    return detections; 
}

cv::Mat YOLOv10::preprocess_image(const cv::Mat& img) {
    int w, h, x, y;
    float r_w = network_width_ / (img.cols*1.0);
    float r_h = network_height_ / (img.rows*1.0);
    if (r_h > r_w) {
        w = network_width_;
        h = r_w * img.rows;
        x = 0;
        y = (network_height_ - h) / 2;
    } else {
        w = r_h * img.cols;
        h = network_height_;
        x = (network_width_ - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(network_width_, network_height_, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    cv::cvtColor(out, out,  cv::COLOR_BGR2RGB);
    out.convertTo(out, CV_32F, 1.0/255.0);
    return out;
}
