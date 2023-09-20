#include "YoloVn.hpp"
YoloVn::YoloVn(
    std::string modelBinary, 
    bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height    
) : 
    net_ {cv::dnn::readNet(modelBinary)}, 
    Yolo{modelBinary, use_gpu, confidenceThreshold,
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


std::vector<Detection> YoloVn::run_detection(const cv::Mat& frame){    
    cv::Mat inputBlob = preprocess_image_mat(frame);
    cv::dnn::blobFromImage(inputBlob, inputBlob, 1 / 255.F, cv::Size(), cv::Scalar(), true, false);
    std::vector<cv::Mat> outs;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
	net_.setInput(inputBlob);
    net_.forward(outs, net_.getUnconnectedOutLayersNames());

    // Resizing factor.
    float x_factor = frame.cols / network_width_;
    float y_factor = frame.rows / network_height_;

    float *data = (float *)outs[0].data;

    int rows = outs[0].size[1];
    int dimensions = outs[0].size[2];
    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= confidenceThreshold_) 
        {
            float * classes_scores = data + 5;
            // Create a 1xDimensions Mat and store class scores of N classes.
            cv::Mat scores(1, dimensions - 5, CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            cv::Point class_id;
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
