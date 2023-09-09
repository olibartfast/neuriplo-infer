#pragma once
#include "Detector.hpp"
#include <NvInfer.h>  // for TensorRT API
#include <cuda_runtime_api.h>  // for CUDA runtime API
#include <fstream>

#include "Logger.hpp"

class RtDetr : public Detector
{
private:
    std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    std::vector<void*> buffers_;
    std::vector<nvinfer1::Dims> output_dims_; // and one output
    std::vector<std::vector<float>> h_outputs_;
    nvinfer1::IRuntime* runtime_{nullptr};

public:
    RtDetr(const std::string& model_path, bool use_gpu = false,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);


    // Create the TensorRT runtime and deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> createRuntimeAndDeserializeEngine(const std::string& engine_path, Logger& logger, nvinfer1::IRuntime*& runtime);

    // Create execution context and allocate input/output buffers
    void createContextAndAllocateBuffers(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext*& context, std::vector<void*>& buffers, std::vector<nvinfer1::Dims>& output_dims, std::vector<std::vector<float>>& h_outputs);

    void initializeBuffers(const std::string& engine_path);

    // calculate size of tensor
    size_t getSizeByDim(const nvinfer1::Dims& dims);


    // Destructor
    ~RtDetr()
    {
        for (void* buffer : buffers_)
        {
            cudaFree(buffer);
        }
    }

    std::vector<Detection> run_detection(const cv::Mat& image) override;     
    std::vector<float> preprocess_image(const cv::Mat& image);
    std::vector<Detection> postprocess(const float* output0, const std::vector<int64_t>& shape0, const cv::Size& frame_size);   
};
 