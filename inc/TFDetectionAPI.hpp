#pragma once
#include "Detector.hpp"
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include "opencv2/opencv.hpp"

class TFDetectionAPI : public Detector{

public:
    TFDetectionAPI(const std::string& model_path, float score_threshold = 0.5f) :
        model_path_(model_path),
        score_threshold_(score_threshold) 
    {
        tensorflow::SessionOptions session_options;
        tensorflow::RunOptions run_options; 
        tensorflow::Status status = LoadSavedModel(session_options, run_options, 
            model_path, {tensorflow::kSavedModelTagServe}, &bundle_);

        if (!status.ok()) {
            std::cout << "Error loading SavedModel: " << status.ToString() << "\n";
            std::exit(1);
        }

        // Create a new session and attach the graph
        session_.reset(bundle_.session.get());

    }

    ~TFDetectionAPI() {
        tensorflow::Status status = session_->Close();
        if (!status.ok()) {
            std::cerr << "Error closing TensorFlow session: " << status.ToString() << std::endl;
        }
    }

   
    std::vector<Detection> run_detection(const cv::Mat& frame) override;

private:
    float compute_iou(const Detection& a, const Detection& b);
    std::vector<int> ApplyNMS(const std::vector<Detection>& detections, float iou_threshold);
    tensorflow::Tensor preprocess(const cv::Mat& frame);

    std::string model_path_;
    float score_threshold_;
    tensorflow::SavedModelBundle bundle_;   
    std::unique_ptr<tensorflow::Session> session_;     
};