#include "YoloVn.hpp"

YoloVn::YoloVn(
    float confidenceThreshold,
    size_t network_width,
    size_t network_height    
) : 
    Detector{confidenceThreshold,
    network_width,
    network_height}
{
}

cv::Mat YoloVn::preprocess_image(const cv::Mat& img) {
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
    cv::cvtColor(out, out,  cv::COLOR_BGR2RGB);
    out.convertTo(out, CV_32F, 1.0/255.0);

    return out;
}


cv::Rect YoloVn::get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
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


std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> YoloVn::postprocess_v567(const TensorElement* output, const std::vector<int64_t>& shape, const cv::Size& frame_size)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    const auto offset = 5;
    const auto num_classes = shape[2] - offset; // 1 x 25200 x 85

    for (int i = 0; i < shape[1]; ++i) {
        const auto obj_conf = std::get<float>(output[4]);
        auto maxSPtr = std::max_element(output + 5, output + 5 + num_classes, 
            [](const TensorElement& a, const TensorElement& b) {
                return std::get<float>(a) < std::get<float>(b);
            });

        float score = std::get<float>(*maxSPtr) * obj_conf;
        if (score > confidenceThreshold_)
        {
            std::vector<float> bbox;
            for (int j = 0; j < 4; ++j) {
                bbox.emplace_back(std::get<float>(output[j]));
            }
            boxes.emplace_back(get_rect(frame_size, bbox));
            int label = maxSPtr - (output + 5);
            confs.emplace_back(score);
            classIds.emplace_back(label);
        }
        output += shape[2];
    }
    return std::make_tuple(boxes, confs, classIds);
}

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> YoloVn::postprocess_ultralytics(const TensorElement* output, const std::vector<int64_t>& shape, const cv::Size& frame_size)
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
            output_matrix[i][j] = std::get<float>(output[i * shape[2] + j]);
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

std::vector<Detection> YoloVn::postprocess(const std::vector<std::vector<TensorElement>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size)
{
    const TensorElement* output0 = outputs.front().data();
    const std::vector<int64_t> shape0 = shapes.front();    

    const auto [boxes, confs, classIds] = (shape0[1] > shape0[2]) ? 
        postprocess_v567(output0, shape0, frame_size) : 
        postprocess_ultralytics(output0, shape0, frame_size); 

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