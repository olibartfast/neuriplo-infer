#include "YoloV8.hpp"

YoloV8::YoloV8(
    std::string modelBinary, 
    bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height    
) : 
    OCVDNNInfer{"", modelBinary, use_gpu, confidenceThreshold,
    network_width,
    network_height}
{


}

cv::Mat YoloV8::preprocess_image(const cv::Mat& img) {
    int w, h, x, y;
    float r_w = network_width_ / (img.cols*1.0);
    float r_h = network_height_ / (img.rows*1.0);
    if (r_h > r_w) {
        w = network_width_;
        h = r_w * img.rows;
        x = 0;
        y = (network_height_ - h) / 2;
    } else {
        w = r_h * img.cols;
        h = network_height_;
        x = (network_width_ - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(network_width_, network_height_, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    cv::dnn::blobFromImage(out, out, 1 / 255.F, cv::Size(), cv::Scalar(), true, false);
    return out;
}



std::vector<Detection> YoloV8::postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size)
{
    const float*  output0 = outputs.front().data();
    const  std::vector<int64_t> shape0 = shapes.front();

    const auto offset = 4;
    const auto num_classes = shape0[1] - offset;
    std::vector<std::vector<float>> output0_matrix(shape0[1], std::vector<float>(shape0[2]));

    // Construct output matrix
    for (size_t i = 0; i < shape0[1]; ++i) {
        for (size_t j = 0; j < shape0[2]; ++j) {
            output0_matrix[i][j] = output0[i * shape0[2] + j];
        }
    }

    std::vector<std::vector<float>> transposed_output0(shape0[2], std::vector<float>(shape0[1]));

    // Transpose output matrix
    for (int i = 0; i < shape0[1]; ++i) {
        for (int j = 0; j < shape0[2]; ++j) {
            transposed_output0[j][i] = output0_matrix[i][j];
        }
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    
    std::vector<std::vector<float>> picked_proposals;

    // Get all the YOLO proposals
    for (int i = 0; i < shape0[2]; ++i) {
        const auto& row = transposed_output0[i];
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

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confidenceThreshold_, nms_threshold_, indices);
    std::vector<Detection> detections;
    for (int i = 0; i < indices.size(); i++)
    {
        Detection det;
        int idx = indices[i];
        det.label = classIds[idx];
        det.bbox = boxes[idx];
        det.score = confs[idx];
        detections.emplace_back(det);
    }
    return detections;
}
