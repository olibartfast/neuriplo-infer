#pragma once
#include "InferenceInterface.hpp"
#include <torch/torch.h>
#include <torch/script.h>

class LibtorchInfer : public InferenceInterface
{
protected:
    torch::DeviceType device_;
    torch::jit::script::Module module_;

public:
    LibtorchInfer(const std::string& model_path, bool use_gpu = true);

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;
  
};