#pragma once
#include "InferenceInterface.hpp"
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include "opencv2/opencv.hpp"

class TFDetectionAPI : public InferenceInterface{

public:
    TFDetectionAPI(const std::string& model_path) :
    InferenceInterface{model_path, "", false}
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

private:

    std::string model_path_;
    tensorflow::SavedModelBundle bundle_;   
    std::unique_ptr<tensorflow::Session> session_; 
    
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;    
};