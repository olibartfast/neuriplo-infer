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
    cv::Mat output_image;
    cv::cvtColor(image, output_image, cv::COLOR_BGR2RGB);
    cv::resize(output_image, output_image, cv::Size(network_width_, network_height_), 0, 0, cv::INTER_LINEAR);
    output_image.convertTo(output_image, CV_32F, 1.0/255.0);    
    return output_image;    
}

std::vector<Detection> YoloNas::postprocess(const std::vector<std::vector<TensorElement>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) 
{
    const TensorElement* output0 = outputs[0].data();
    const std::vector<int64_t> shape0 = shapes[0];

    const TensorElement* output1 = outputs[1].data();
    const std::vector<int64_t> shape1 = shapes[1];

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // idx 0 boxes, idx 1 scores
    int rows = shape0[1]; // 8400
    int dimensions_boxes = shape0[2];  // 4
    int dimensions_scores = shape1[2]; // num classes (80)

    float r_w = static_cast<float>(frame_size.width) / network_width_;
    float r_h = static_cast<float>(frame_size.height) / network_height_;

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        auto maxSPtr = std::max_element(output1, output1 + dimensions_scores, 
            [](const TensorElement& a, const TensorElement& b) {
                return std::get<float>(a) < std::get<float>(b);
            });

        float score = std::get<float>(*maxSPtr);
        if (score >= confidenceThreshold_) 
        {
            int label = static_cast<int>(maxSPtr - output1);
            confidences.push_back(score);
            classIds.push_back(label);

            std::vector<float> bbox;
            for(int j = 0; j < 4; j++) {
                bbox.emplace_back(std::get<float>(*(output0 + j)));
            }

            // Ensure coordinates are within frame boundaries
            int left = std::max(0, static_cast<int>(bbox[0] * r_w));
            int top = std::max(0, static_cast<int>(bbox[1] * r_h));
            int width = std::min(frame_size.width - left, 
                               static_cast<int>((bbox[2] - bbox[0]) * r_w));
            int height = std::min(frame_size.height - top, 
                                static_cast<int>((bbox[3] - bbox[1]) * r_h));
            
            boxes.emplace_back(left, top, width, height);
        }
        // Jump to the next column
        output1 += dimensions_scores;
        output0 += dimensions_boxes;
    }

    // Perform Non Maximum Suppression and draw predictions
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold_, nms_threshold_, indices);
    
    std::vector<Detection> detections;
    detections.reserve(indices.size());
    
    for (int idx : indices) 
    {
        Detection det{
            boxes[idx],       // bbox
            confidences[idx], // score
            classIds[idx]     // label
        };
        detections.emplace_back(det);
    }
    
    return detections; 
}