#include "YoloV4.hpp"

    YoloV4::YoloV4(
        float confidenceThreshold,
        size_t network_width,
        size_t network_height    
    ) : 

        Detector{
        confidenceThreshold,
        network_width,
        network_height}
	{

	}


cv::Mat YoloV4::preprocess_image(const cv::Mat& image)
{
    cv::Mat output_image;
    cv::cvtColor(image, output_image, cv::COLOR_BGR2RGB);
    cv::resize(output_image, output_image, cv::Size(network_width_, network_height_));
    output_image.convertTo(output_image, CV_32F, 1.0/255.0);
    return output_image;    
}


std::vector<Detection> YoloV4::postprocess(const std::vector<std::vector<std::any>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    const auto cols = frame_size.width;
    const auto rows = frame_size.height;

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        const std::any* output = outputs[i].data();
        for (int j = 0; j < shapes[i][0]; ++j, output += shapes[i][1])
        {
            auto maxSPtr = std::max_element(output + 5, output + shapes[i][1], [](const std::any& a, const std::any& b) {
                return std::any_cast<float>(a) < std::any_cast<float>(b);
            });
            float score = std::any_cast<float>(*maxSPtr);
            if (score > confidenceThreshold_)
            {
                int centerX = std::any_cast<float>(output[0]) * cols;
                int centerY = std::any_cast<float>(output[1]) * rows;
                int width = std::any_cast<float>(output[2]) * cols;
                int height = std::any_cast<float>(output[3]) * rows;
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                int label = maxSPtr - (output + 5);
                classIds.push_back(label);
                confidences.push_back(score);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<Detection> detections;
    std::map<int, std::vector<size_t> > class2indices;
    for (size_t i = 0; i < classIds.size(); i++)
    {
        if (confidences[i] >= confidenceThreshold_)
        {
            class2indices[classIds[i]].push_back(i);
        }
    }

    for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
    {
        std::vector<cv::Rect> localBoxes;
        std::vector<float> localConfidences;
        std::vector<size_t> classIndices = it->second;
        for (size_t i = 0; i < classIndices.size(); i++)
        {
            localBoxes.push_back(boxes[classIndices[i]]);
            localConfidences.push_back(confidences[classIndices[i]]);
        }
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(localBoxes, localConfidences, confidenceThreshold_, nms_threshold_, nmsIndices);
        for (size_t i = 0; i < nmsIndices.size(); i++)
        {
            Detection d;
            size_t idx = nmsIndices[i];
            d.bbox = localBoxes[idx];
            d.score = localConfidences[idx];
            d.label = it->first;
            detections.emplace_back(d);

        }
    }

    return detections;
}