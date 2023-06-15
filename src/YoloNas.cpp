#include "YoloNas.hpp"

YoloNas::YoloNas(
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


std::vector<Detection> YoloNas::run_detection(const Mat& frame){    
    cv::Mat inputBlob;
    cv::dnn::blobFromImage(frame, inputBlob, 1 / 255.F, cv::Size(network_width_, network_height_), cv::Scalar(), true, false, CV_32F);
    std::vector<cv::Mat> outs;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
	net_.setInput(inputBlob);
    net_.forward(outs, net_.getUnconnectedOutLayersNames());

    float *scores_data = (float *)outs[0].data;
    float *boxes_data = (float *)outs[1].data;

    int rows = outs[0].size[1];
    int dimensions_scores = outs[0].size[2];
    int dimensions_boxes = outs[1].size[2];

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        cv::Mat scores(1, classNames_.size(), CV_32FC1, scores_data);
        cv::Point class_id;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if (maxClassScore >= confidenceThreshold_) 
        {
            confidences.push_back(maxClassScore);
            classIds.push_back(class_id.x);
            float r_w = (frame.cols * 1.0) / network_width_;
            float r_h = (frame.rows * 1.0) / network_height_ ;
            std::vector<float> bbox(&boxes_data[0], &boxes_data[4]);

            int left = (int)(bbox[0] * r_w);
            int top = (int)(bbox[1] * r_h);
            int width = (int)((bbox[2] - bbox[0]) * r_w);
            int height = (int)((bbox[3] - bbox[1]) * r_h);
            boxes.push_back(Rect(left, top, width, height));
        }
        // Jump to the next column.
        scores_data += dimensions_scores;
        boxes_data += dimensions_boxes;
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

