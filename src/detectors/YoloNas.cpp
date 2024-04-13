#include "YoloNas.hpp"

YoloNas::YoloNas(
    float confidenceThreshold,
    size_t network_width,
    size_t network_height    
) : 
    Detector{confidenceThreshold,
    network_width,
    network_height}
{


}



cv::Mat YoloNas::preprocess_image(const cv::Mat& image)
{
    cv::Mat blob;
    cv::cvtColor(image, blob, cv::COLOR_BGR2RGB);
    cv::Mat resized_image(network_height_, network_width_, CV_8UC3);
    cv::resize(blob, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);
    cv::dnn::blobFromImage(resized_image, resized_image, 1 / 255.F, cv::Size(), cv::Scalar(), true, false);      
    return resized_image;    
}



std::vector<Detection> YoloNas::postprocess(const std::vector<std::vector<std::any>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) 
{
    const float*  output0 = outputs[0].data();
    const  std::vector<int64_t> shape0 = shapes[0];

    const float*  output1 = outputs[1].data();
    const  std::vector<int64_t> shape1 = shapes[1];

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // idx 0 boxes, idx 1 scores
    int rows = shape0[1]; // 8400
    int dimensions_boxes = shape0[2];  // 4
    int dimensions_scores = shape1[2]; // num classes (80)

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        auto maxSPtr = std::max_element(output1, output1 + dimensions_scores);
        float score = *maxSPtr;
        if (score >= confidenceThreshold_) 
        {
            int label = maxSPtr - output1;
            confidences.push_back(score);
            classIds.push_back(label);
            float r_w = (frame_size.width * 1.0) / network_width_;
            float r_h = (frame_size.height * 1.0) / network_height_ ;
            std::vector<float> bbox(&output0[0], &output0[4]);

            int left = (int)(bbox[0] * r_w);
            int top = (int)(bbox[1] * r_h);
            int width = (int)((bbox[2] - bbox[0]) * r_w);
            int height = (int)((bbox[3] - bbox[1]) * r_h);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        // Jump to the next column.
        output1 += dimensions_scores;
        output0 += dimensions_boxes;
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