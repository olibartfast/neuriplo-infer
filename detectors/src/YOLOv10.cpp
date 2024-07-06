
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
            float r_w = (frame_size.width * 1.0) / network_width_;
            float r_h = (frame_size.height * 1.0) / network_height_ ;

            float x1 = std::any_cast<float>(*output0) * r_w;
            float y1 = std::any_cast<float>(*(output0 + 1)) * r_h;
            float x2 = std::any_cast<float>(*(output0 + 2)) * r_w;
            float y2 = std::any_cast<float>(*(output0 + 3)) * r_h;

            det.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            detections.emplace_back(det);
        }
        output0 += shape0[2];
    }
    return detections; 
}

cv::Mat YOLOv10::preprocess_image(const cv::Mat& image)
{
    cv::Mat output_image;   
    cv::dnn::blobFromImage(image, output_image, 1.f / 255.f, cv::Size(network_height_, network_width_), cv::Scalar(), true, false);
    return output_image;
}