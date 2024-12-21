#include "DFine.hpp"

DFine::DFine(
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{confidenceThreshold,
            network_width,
            network_height}
{

}

std::vector<Detection> DFine::postprocess(
    const std::vector<std::vector<TensorElement>>& outputs, 
    const std::vector<std::vector<int64_t>>& shapes, 
    const cv::Size& frame_size) {

    size_t labels_idx = 0;
    size_t boxes_idx = 1;
    size_t scores_idx = 2;
    
    const auto& scores = outputs[scores_idx];
    const std::vector<int64_t>& shape_scores = shapes[scores_idx];

    const auto& boxes = outputs[boxes_idx];
    const std::vector<int64_t>& shape_boxes = shapes[boxes_idx];

    const auto& labels = outputs[labels_idx];
    const std::vector<int64_t>& shape_labels = shapes[labels_idx];

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes_rect;

    int rows = shape_labels[1]; // 300

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) {
        float score = std::get<float>(scores[i]);
        if (score >= confidenceThreshold_) {
            // Handle label using std::visit
            std::visit([&classIds](auto&& label) {
                using T = std::decay_t<decltype(label)>;
                if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
                    classIds.push_back(static_cast<int>(label));
                }
            }, labels[i]);

            confidences.push_back(score);
            float r_w = (float)frame_size.width / network_width_;
            float r_h = (float)frame_size.height / network_height_;
            
            float x1 = std::get<float>(boxes[i*4]);
            float y1 = std::get<float>(boxes[i*4 + 1]);
            float x2 = std::get<float>(boxes[i*4 + 2]);
            float y2 = std::get<float>(boxes[i*4 + 3]);
            
            x2 *= r_w;
            y2 *= r_h;
            x1 *= r_w;
            y1 *= r_h;
            boxes_rect.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_rect, confidences, confidenceThreshold_, nms_threshold_, indices);
    
    std::vector<Detection> detections;
    for (int i = 0; i < indices.size(); i++) 
    {
        Detection det;
        int idx = indices[i];
        det.label = classIds[idx];
        det.bbox = boxes_rect[idx];
        det.score = confidences[idx];
        detections.emplace_back(det);
    }
    return detections; 
}

// The preprocess_image function remains unchanged
cv::Mat DFine::preprocess_image(const cv::Mat& image)
{
    cv::Mat output_image;   
    cv::cvtColor(image, output_image,  cv::COLOR_BGR2RGB);
    cv::resize(output_image, output_image, cv::Size(network_width_, network_height_));
    output_image.convertTo(output_image, CV_32F, 1.0/255.0);
    return output_image;
}
