
#include "RtDetrUltralytics.hpp"

RtDetrUltralytics::RtDetrUltralytics(
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{confidenceThreshold,
            network_width,
            network_height}
{

}


std::vector<Detection> RtDetrUltralytics::postprocess(const std::vector<std::vector<std::any>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) 
{
    const std::any* output0 = outputs.front().data();
    const  std::vector<int64_t> shape0 = shapes.front();

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // idx 0 boxes, idx 1 scores
    int rows = shape0[1]; // 300
    int dimensions_scores = shape0[2] - 4; // num classes (80)

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        auto maxSPtr = std::max_element(output0 + 4 , output0 + 4 + dimensions_scores, [](const std::any& a, const std::any& b) {
            return std::any_cast<float>(a) < std::any_cast<float>(b);
        });

        float score = std::any_cast<float>(*maxSPtr);
        if (score >= confidenceThreshold_) 
        {
            int label = maxSPtr - output0 - 4;
            confidences.push_back(score);
            classIds.push_back(label);
            float r_w = frame_size.width;
            float r_h = frame_size.height;

            float b0 = std::any_cast<float>(*output0);
            float b1 = std::any_cast<float>(*(output0 + 1));
            float b2 = std::any_cast<float>(*(output0 + 2));
            float b3 = std::any_cast<float>(*(output0 + 3));

            float x1 = b0 - b2 / 2.0f;
            float y1 = b1 - b3 / 2.0f;
            float x2 = b0 + b2 / 2.0f;
            float y2 = b1 + b3 / 2.0f;
            x2 *= r_w;
            y2 *= r_h;
            x1 *= r_w;
            y1 *= r_h;
            boxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
        output0 += shape0[2];
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

cv::Mat RtDetrUltralytics::preprocess_image(const cv::Mat& image)
{
    cv::Mat output_image;   
    cv::dnn::blobFromImage(image, output_image, 1.f / 255.f, cv::Size(network_height_, network_width_), cv::Scalar(), true, false);
    return output_image;
}