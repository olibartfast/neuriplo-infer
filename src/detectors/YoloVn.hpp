#pragma once
#include "Detector.hpp"
class YoloVn : public Detector{ 

public:
    YoloVn(
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);    
        
    std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) override;
    cv::Mat preprocess_image(const cv::Mat& image) override; 

    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
    {
        float r_w = network_width_ / static_cast<float>(imgSz.width);
        float r_h = network_height_ / static_cast<float>(imgSz.height);
        
        int l, r, t, b;
        if (r_h > r_w) {
            l = bbox[0] - bbox[2] / 2.f;
            r = bbox[0] + bbox[2] / 2.f;
            t = bbox[1] - bbox[3] / 2.f - (network_height_ - r_w * imgSz.height) / 2;
            b = bbox[1] + bbox[3] / 2.f - (network_height_ - r_w * imgSz.height) / 2;
            l /= r_w;
            r /= r_w;
            t /= r_w;
            b /= r_w;
        }
        else {
            l = bbox[0] - bbox[2] / 2.f - (network_width_ - r_h * imgSz.width) / 2;
            r = bbox[0] + bbox[2] / 2.f - (network_width_ - r_h * imgSz.width) / 2;
            t = bbox[1] - bbox[3] / 2.f;
            b = bbox[1] + bbox[3] / 2.f;
            l /= r_h;
            r /= r_h;
            t /= r_h;
            b /= r_h;
    }

        // Clamp the coordinates within the image bounds
        l = std::max(0, std::min(l, imgSz.width - 1));
        r = std::max(0, std::min(r, imgSz.width - 1));
        t = std::max(0, std::min(t, imgSz.height - 1));
        b = std::max(0, std::min(b, imgSz.height - 1));

        return cv::Rect(l, t, r - l, b - t);
    }


    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> postprocess_v567(const float* output, const std::vector<int64_t>& shape, const cv::Size& frame_size)
    {
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;
 
        const auto offset = 5;
        const auto num_classes = shape[2] - offset; // 1 x 25200 x 85

        // Get all the YOLO proposals
        for (int i = 0; i < shape[1]; ++i) {
            if(output[4] > confidenceThreshold_)
            {
                const float* scoresPtr = output + 5;
                auto maxSPtr = std::max_element(scoresPtr, scoresPtr + num_classes);
                float score = *maxSPtr * output[4];
                if (score > confidenceThreshold_) {
                    boxes.emplace_back(get_rect(frame_size, std::vector<float>(output, output + 4)));
                    int label = maxSPtr - scoresPtr;
                    confs.emplace_back(score);
                    classIds.emplace_back(label);
                }

            }
            output += shape[2]; 
        }
        return std::make_tuple(boxes, confs, classIds);
    }


    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> postprocess_v89(const float* output, const std::vector<int64_t>& shape, const cv::Size& frame_size)
    {
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;


        const auto offset = 4;
        const auto num_classes = shape[1] - offset;
        std::vector<std::vector<float>> output_matrix(shape[1], std::vector<float>(shape[2]));

        // Construct output matrix
        for (size_t i = 0; i < shape[1]; ++i) {
            for (size_t j = 0; j < shape[2]; ++j) {
                output_matrix[i][j] = output[i * shape[2] + j];
            }
        }

        std::vector<std::vector<float>> transposed_output(shape[2], std::vector<float>(shape[1]));

        // Transpose output matrix
        for (int i = 0; i < shape[1]; ++i) {
            for (int j = 0; j < shape[2]; ++j) {
                transposed_output[j][i] = output_matrix[i][j];
            }
        }

        // Get all the YOLO proposals
        for (int i = 0; i < shape[2]; ++i) {
            const auto& row = transposed_output[i];
            const float* bboxesPtr = row.data();
            const float* scoresPtr = bboxesPtr + 4;
            auto maxSPtr = std::max_element(scoresPtr, scoresPtr + num_classes);
            float score = *maxSPtr;
            if (score > confidenceThreshold_) {
                boxes.emplace_back(get_rect(frame_size, std::vector<float>(bboxesPtr, bboxesPtr + 4)));
                int label = maxSPtr - scoresPtr;
                confs.emplace_back(score);
                classIds.emplace_back(label);
            }
        }
        return std::make_tuple(boxes, confs, classIds);
    }
};