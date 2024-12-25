#include "YoloV4.hpp"
YoloV4::YoloV4(const ModelInfo& model_info, float confidenceThreshold) : Detector{model_info, confidenceThreshold}
{

}

cv::Mat YoloV4::preprocess_image(const cv::Mat& image)
{
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    
    cv::Mat resized_image;
    cv::resize(rgb_image, resized_image, cv::Size(network_width_, network_height_));
    
    cv::Mat normalized_image;
    resized_image.convertTo(normalized_image, CV_32F, 1.0/255.0);
    
    return normalized_image;
}

std::vector<Detection> YoloV4::postprocess(const std::vector<std::vector<TensorElement>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size)
{
    std::vector<Detection> detections;
    
    // Iterate over the output tensors
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        const TensorElement* output = outputs[i].data();
        
        // Iterate over the detections in the output tensor
        for (int j = 0; j < shapes[i][0]; ++j, output += shapes[i][1])
        {
            // Find the class with the highest confidence
            auto maxSPtr = std::max_element(output + 5, output + shapes[i][1], [](const TensorElement& a, const TensorElement& b) {
                return std::get<float>(a) < std::get<float>(b);
            });
            float score = std::get<float>(*maxSPtr);
            
            // Check if the confidence is above the threshold
            if (score > confidenceThreshold_)
            {
                int centerX = std::get<float>(output[0]) * frame_size.width;
                int centerY = std::get<float>(output[1]) * frame_size.height;
                int width = std::get<float>(output[2]) * frame_size.width;
                int height = std::get<float>(output[3]) * frame_size.height;
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                int label = maxSPtr - (output + 5);
                
                // Create a detection object
                Detection detection;
                detection.bbox = cv::Rect(left, top, width, height);
                detection.score = score;
                detection.label = label;
                
                // Add the detection to the vector
                detections.push_back(detection);
            }
        }
    }
    
    // Apply non-maximum suppression to the detections
    std::vector<Detection> filtered_detections;
    std::map<int, std::vector<size_t> > class2indices;
    for (size_t i = 0; i < detections.size(); i++)
    {
        if (detections[i].score >= confidenceThreshold_)
        {
            class2indices[detections[i].label].push_back(i);
        }
    }
    
    for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
    {
        std::vector<cv::Rect> localBoxes;
        std::vector<float> localConfidences;
        std::vector<size_t> classIndices = it->second;
        for (size_t i = 0; i < classIndices.size(); i++)
        {
            localBoxes.push_back(detections[classIndices[i]].bbox);
            localConfidences.push_back(detections[classIndices[i]].score);
        }
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(localBoxes, localConfidences, confidenceThreshold_, nms_threshold_, nmsIndices);
        for (size_t i = 0; i < nmsIndices.size(); i++)
        {
            filtered_detections.push_back(detections[classIndices[nmsIndices[i]]]);
        }
    }
    
    return filtered_detections;
}