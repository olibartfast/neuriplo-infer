#include "Yolo.hpp"

Yolo::Yolo(const std::string& model_path, bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{model_path, use_gpu, confidenceThreshold,
            network_width,
            network_height}
{

}

cv::Rect Yolo::get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
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


std::vector<float> Yolo::preprocess_image(const cv::Mat &image)
{
    cv::Mat blob;
    cv::cvtColor(image, blob, cv::COLOR_BGR2RGB);
    int target_width, target_height, offset_x, offset_y;
    float resize_ratio_width = static_cast<float>(network_width_) / static_cast<float>(image.cols);
    float resize_ratio_height = static_cast<float>(network_height_) / static_cast<float>(image.rows);

    if (resize_ratio_height > resize_ratio_width)
    {
        target_width = network_width_;
        target_height = resize_ratio_width * image.rows;
        offset_x = 0;
        offset_y = (network_height_ - target_height) / 2;
    }
    else
    {
        target_width = resize_ratio_height * image.cols;
        target_height = network_height_;
        offset_x = (network_width_ - target_width) / 2;
        offset_y = 0;
    }

    cv::Mat resized_image(target_height, target_width, CV_8UC3);
    cv::resize(blob, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat output_image(network_width_, network_height_, CV_8UC3, cv::Scalar(128, 128, 128));
    resized_image.copyTo(output_image(cv::Rect(offset_x, offset_y, resized_image.cols, resized_image.rows)));   
    output_image.convertTo(output_image, CV_32FC3, 1.f / 255.f);        

    size_t img_byte_size = output_image.total() * output_image.elemSize();  // Allocate a buffer to hold all image elements.
    std::vector<float> input_data = std::vector<float>(network_width_ * network_height_ * channels_);
    std::memcpy(input_data.data(), output_image.data, img_byte_size);

    std::vector<cv::Mat> chw;
    for (size_t i = 0; i < channels_; ++i)
    {
        chw.emplace_back(cv::Mat(cv::Size(network_width_, network_height_), CV_32FC1, &(input_data[i * network_width_ * network_height_])));
    }
    cv::split(output_image, chw);

    return input_data;
}


std::vector<Detection> Yolo::postprocess(const float*  output0, const  std::vector<int64_t>& shape0,  const cv::Size& frame_size)
{

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

cv::Mat Yolo::preprocess_image_mat(const cv::Mat& img) {
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
    return out;
}



std::vector<Detection> Yolo::postprocess(const float* output0, const float* output1 ,const std::vector<int64_t>& shape0, const std::vector<int64_t>& shape1, const cv::Size& frame_size)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // idx 0 boxes, idx 1 scores
    int rows = shape0[1]; // 8400
    int dimensions_boxes = shape0[2];  // 4
    int dimensions_scores = shape1[2]; // num classes (80)

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        auto maxSPtr = std::max_element(output1, output1 + dimensions_scores);
        float score = *maxSPtr;
        if (score >= confidenceThreshold_) 
        {
            int label = maxSPtr - output1;
            confidences.push_back(score);
            classIds.push_back(label);
            float r_w = (frame_size.width * 1.0) / network_width_;
            float r_h = (frame_size.height * 1.0) / network_height_ ;
            std::vector<float> bbox(&output0[0], &output0[4]);

            int left = (int)(bbox[0] * r_w);
            int top = (int)(bbox[1] * r_h);
            int width = (int)((bbox[2] - bbox[0]) * r_w);
            int height = (int)((bbox[3] - bbox[1]) * r_h);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        // Jump to the next column.
        output1 += dimensions_scores;
        output0 += dimensions_boxes;
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