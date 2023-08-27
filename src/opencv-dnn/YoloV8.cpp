#include "YoloV8.hpp"

YoloV8::YoloV8(std::string modelBinary, 
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

std::vector<Detection> YoloV8::run_detection(const cv::Mat& frame){    
    cv::Mat inputPreprocessed =  preprocess_image_mat(frame);
    cv::Mat inputBlob;
    cv::dnn::blobFromImage(inputPreprocessed, inputBlob, 1 / 255.F, cv::Size(inputPreprocessed.rows, inputPreprocessed.cols), cv::Scalar(), true, false);
    std::vector<cv::Mat> outs;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
	net_.setInput(inputBlob);
    net_.forward(outs, net_.getUnconnectedOutLayersNames());

    // Resizing factor.
    float x_factor = frame.cols / network_width_;
    float y_factor = frame.rows / network_height_;



    int rows = outs[0].size[1];
    int dimensions = outs[0].size[2];

    if (dimensions > rows) 
    {
        rows = outs[0].size[2];
        dimensions = outs[0].size[1];

        outs[0] = outs[0].reshape(1, dimensions);
        cv::transpose(outs[0], outs[0]);
    }   

    float *data = (float *)outs[0].data;
  
    for (int i = 0; i < rows; ++i)
    {
        float *classes_scores = data+4;

        cv::Mat scores(1, dimensions-4, CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > confidenceThreshold_)
        {
            confidences.push_back(maxClassScore);
            classIds.push_back(class_id.x);
            std::vector<float> bbox(&data[0], &data[4]);
            cv::Rect r = get_rect(frame.size(), bbox);

            boxes.push_back(r);
        }
        data += dimensions;
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