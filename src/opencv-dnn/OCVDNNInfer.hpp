#pragma once
#include "InferenceEngine.hpp"

class OCVDNNInfer : public InferenceEngine
{
private:
	cv::dnn::Net net_;
    std::vector<int> outLayers_;
    std::string outLayerType_;
    std::vector<std::string> outNames_;
        
public:
    OCVDNNInfer(const std::string& weights, const std::string& modelConfiguration = "");

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;
};
