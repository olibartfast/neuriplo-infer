#include "RtDetr.hpp"

RtDetr::RtDetr(
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{confidenceThreshold,
            network_width,
            network_height}
{

}


std::vector<Detection> RtDetr::postprocess(const std::vector<std::vector<std::any>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) {

    size_t labels_idx = 0;
    size_t boxes_idx = 1;
    size_t scores_idx = 2;

    // Output order of this model somewhat changes when it is export to TensorRT.
    if(shapes[2][2] == 4)
    {
        labels_idx = 1;
        boxes_idx = 2;
        scores_idx = 0;
    }

    const std::any* scores_ptr = outputs[scores_idx].data();
    const std::vector<int64_t>& shape_scores = shapes[scores_idx];

    const std::any* boxes_ptr = outputs[boxes_idx].data();
    const std::vector<int64_t>& shape_boxes = shapes[boxes_idx];

    const std::any* labels_ptr = outputs[labels_idx].data();
    const std::vector<int64_t>& shape_labels = shapes[labels_idx];


    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    int rows = shape_labels[1]; // 300

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) {
        float score = std::any_cast<float>(scores_ptr[i]);
        if (score >= confidenceThreshold_) {
            auto label = std::any_cast<int>(labels_ptr[i]);
            confidences.push_back(score);
            classIds.push_back(label);
            float r_w = (float)frame_size.width / network_width_;
            float r_h = (float)frame_size.height / network_height_;
            float x1 =  std::any_cast<float>(boxes_ptr[i*4]);
            float y1 =  std::any_cast<float>(boxes_ptr[i*4 + 1]);
            float x2 =  std::any_cast<float>(boxes_ptr[i*4 + 2]);
            float y2 =  std::any_cast<float>(boxes_ptr[i*4 + 3]);
            x2 *= r_w;
            y2 *= r_h;
            x1 *= r_w;
            y1 *= r_h;
            boxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold_, nms_threshold_, indices);
    std::vector<Detection> detections;
    for (int i = 0; i < indices.size(); i++) 
    {
        Detection det;
        int idx = indices[i];
        det.label = classIds[idx];
        det.bbox = boxes[idx];
        det.score = confidences[idx];
        detections.emplace_back(det);
    }
    return detections; 

}

cv::Mat RtDetr::preprocess_image(const cv::Mat& image)
{
    cv::Mat output_image;   
    cv::dnn::blobFromImage(image, output_image, 1.f / 255.f, cv::Size(network_height_, network_width_), cv::Scalar(), true, false);
    return output_image;
}