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



std::vector<Detection> RtDetr::postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) 
{
    const float*  output0 = outputs.front().data();
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
        auto maxSPtr = std::max_element(output0 + 4 , output0 +  4 +dimensions_scores);
        float score = *maxSPtr;
        if (score >= 0.45) 
        {
            int label = maxSPtr - output0 - 4;
            confidences.push_back(score);
            classIds.push_back(label);
            float r_w = frame_size.width;
            float r_h = frame_size.height;
            std::vector<float> bbox(&output0[0], &output0[4]);

            float x1 = bbox[0] -bbox[2] / 2.0f;
            float y1 = bbox[1] - bbox[3] / 2.0f;
            float x2 = bbox[0] + bbox[2] / 2.0f;
            float y2 =bbox[1] + bbox[3] / 2.0f;
            x2 *= r_w;
            y2 *= r_h;
            x1 *= r_w;
            y1 *= r_h;
            boxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
        output0 += shape0[2] ;
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold_, nms_threshold_, indices);
    std::vector<Detection> detections;
    for (int i = 0; i < boxes.size(); i++) 
    {
        Detection det;
        int idx = i;
        det.label = classIds[idx];
        det.bbox = boxes[idx];
        det.score = confidences[idx];
        detections.emplace_back(det);
    }
    return detections; 
}


cv::Mat RtDetr::preprocess_image(const cv::Mat& image)
{
    cv::Mat blob;
    cv::cvtColor(image, blob, cv::COLOR_BGR2RGB);
    cv::Mat resized_image(network_height_, network_width_, CV_8UC3);
    cv::resize(blob, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat output_image;
    resized_image.convertTo(output_image, CV_32FC3, 1.f / 255.f);        

    // size_t img_byte_size = output_image.total() * output_image.elemSize();  // Allocate a buffer to hold all image elements.
    // std::vector<float> input_data = std::vector<float>(network_width_ * network_height_ * channels_);
    // std::memcpy(input_data.data(), output_image.data, img_byte_size);

    // std::vector<cv::Mat> chw;
    // for (size_t i = 0; i < channels_; ++i)
    // {
    //     chw.emplace_back(cv::Mat(cv::Size(network_width_, network_height_), CV_32FC1, &(input_data[i * network_width_ * network_height_])));
    // }
    // cv::split(output_image, chw);

    // return input_data;    
    return output_image;
}