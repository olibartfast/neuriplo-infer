
#include "RtDetrUltralytics.hpp"

RtDetrUltralytics::RtDetrUltralytics(const ModelInfo& model_info, float confidenceThreshold) : Detector{model_info, confidenceThreshold}
{

}


std::vector<Detection> RtDetrUltralytics::postprocess(const std::vector<std::vector<TensorElement>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) 
{
    const TensorElement* output0 = outputs.front().data();
    const  std::vector<int64_t> shape0 = shapes.front();

    std::vector<Detection> detections;

    // idx 0 boxes, idx 1 scores
    int rows = shape0[1]; // 300
    int dimensions_scores = shape0[2] - 4; // num classes (80)

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        auto maxSPtr = std::max_element(output0 + 4 , output0 + 4 + dimensions_scores, [](const TensorElement& a, const TensorElement& b) {
            return std::get<float>(a) < std::get<float>(b);
        });

        float score = std::get<float>(*maxSPtr);
        if (score >= confidenceThreshold_) 
        {
            Detection det;
            det.label = maxSPtr - output0 - 4;
            det.score = score;
            float r_w = frame_size.width;
            float r_h = frame_size.height;

            float b0 = std::get<float>(*output0);
            float b1 = std::get<float>(*(output0 + 1));
            float b2 = std::get<float>(*(output0 + 2));
            float b3 = std::get<float>(*(output0 + 3));

            float x1 = b0 - b2 / 2.0f;
            float y1 = b1 - b3 / 2.0f;
            float x2 = b0 + b2 / 2.0f;
            float y2 = b1 + b3 / 2.0f;
            x2 *= r_w;
            y2 *= r_h;
            x1 *= r_w;
            y1 *= r_h;
            det.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            
            detections.emplace_back(det);
        }
        output0 += shape0[2];
    }
    return detections; 
}

cv::Mat RtDetrUltralytics::preprocess_image(const cv::Mat& image)
{
    cv::Mat output_image;   
    cv::cvtColor(image, output_image,  cv::COLOR_BGR2RGB);
    cv::resize(output_image, output_image, cv::Size(network_width_, network_height_));
    output_image.convertTo(output_image, CV_32F, 1.0/255.0);
    return output_image;
}