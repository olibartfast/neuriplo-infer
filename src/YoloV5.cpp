#include "YoloV5.hpp"

YoloV5::YoloV5(
    const std::vector<std::string>& classNames,
    std::string modelBinary, 
    float confidenceThreshold,
    size_t network_width,
    size_t network_height    
) : 
    net_ {cv::dnn::readNet(modelBinary)}, 
    Detector{classNames, 
    modelBinary, confidenceThreshold,
    network_width,
    network_height}
{
    if (net_.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "weights-file: " << modelBinary << std::endl;
        exit(-1);
    }

}

cv::Mat YoloV5::preprocess_img(const cv::Mat& img) {
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

cv::Rect YoloV5::get_rect(const cv::Size& imgSize, const std::vector<float>& bbox) 
{
    float l, r, t, b;
    float r_w = network_width_/ (imgSize.width * 1.0);
    float r_h = network_height_ / (imgSize.height * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (network_height_- r_w * imgSize.height) / 2;
        b = bbox[1] + bbox[3] / 2.f - (network_height_ - r_w * imgSize.height) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (network_width_- r_h * imgSize.width) / 2;
        r = bbox[0] + bbox[2] / 2.f - (network_width_- r_h * imgSize.width) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

std::vector<Detection> YoloV5::run_detection(const Mat& frame){    
    cv::Mat inputBlob = preprocess_img(frame);
    cv::dnn::blobFromImage(inputBlob, inputBlob, 1 / 255.F, cv::Size(), cv::Scalar(), true, false);
    std::vector<cv::Mat> outs;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
	net_.setInput(inputBlob);
    net_.forward(outs, net_.getUnconnectedOutLayersNames());

    // Resizing factor.
    float x_factor = frame.cols / network_width_;
    float y_factor = frame.rows / network_height_;

    float *data = (float *)outs[0].data;

    int rows = outs[0].size[1];
    int dimensions = outs[0].size[2];
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= confidenceThreshold_) 
        {
            float * classes_scores = data + 5;
            // Create a 1xDimensions Mat and store class scores of N classes.
            cv::Mat scores(1, classNames_.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > score_threshold_) 
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                std::vector<float> bbox(&data[0], &data[4]);
                confidences.push_back(confidence);
                classIds.push_back(class_id.x);
                cv::Rect r = get_rect(frame.size(), bbox);

                // Store good detections in the boxes vector.
                boxes.push_back(r);
            }

        }
        // Jump to the next column.
        data += dimensions;
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);
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
