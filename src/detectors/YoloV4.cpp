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
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1 / 255.F, cv::Size(network_width_, network_height_), cv::Scalar(), true, false, CV_32F);   
    return blob;    
}


std::vector<Detection> YoloV4::postprocess(const std::vector<std::vector<std::any>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size)
{
// outs[0].rows shapes[0][0]
// 1083
// outs[1].rows shapes[0][1]
// 4332
// outs[1].cols shapes[1][1]
// 85
// outs[0].cols shapes[0][1]
// 85

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
        const float* data = (float*)outputs[i].data();
        for (int j = 0; j < shapes[i][0]; ++j, data += shapes[i][1])
        {
            const float* scoresPtr = data + 5;
            auto maxSPtr = std::max_element(scoresPtr, scoresPtr + shapes[i][1] - 5);
            float score = *maxSPtr;
            if (score > confidenceThreshold_)
            {
                int centerX = (int)(data[0] * cols);
                int centerY = (int)(data[1] * rows);
                int width = (int)(data[2] * cols);
                int height = (int)(data[3] * rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                int label = maxSPtr - scoresPtr;
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