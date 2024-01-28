#pragma once
#include "InferenceInterface.hpp"
#include <openvino/openvino.hpp>


class OVInfer : public InferenceInterface
{
protected:


public:
    OVInfer(const std::string& model_path = "", const std::string& modelConfiguration = "", bool use_gpu = true);

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;
  
    ov::Core core_;
    ov::Tensor input_tensor_;
    ov::InferRequest infer_request_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
};