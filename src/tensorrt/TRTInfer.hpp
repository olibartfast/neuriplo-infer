#pragma once
#include "Detector.hpp"
#include <NvInfer.h>  // for TensorRT API
#include <cuda_runtime_api.h>  // for CUDA runtime API
#include <fstream>

#include "Logger.hpp"

class TRTInfer : public Detector
{
    protected:
        std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
        nvinfer1::IExecutionContext* context_{nullptr};
        std::vector<void*> buffers_;
        std::vector<std::vector<int64_t>> output_shapes_;
        std::vector<std::vector<float>> h_outputs_;
        nvinfer1::IRuntime* runtime_{nullptr};

    public:
        TRTInfer(const std::string& model_path, bool use_gpu = true,
        float confidenceThreshold = 0.25,
        size_t network_width = 640,
        size_t network_height = 640);

        // Create execution context and allocate input/output buffers
        void createContextAndAllocateBuffers();

        void initializeBuffers(const std::string& engine_path);

        // calculate size of tensor
        size_t getSizeByDim(const nvinfer1::Dims& dims);    

        void infer();

        std::vector<Detection> run_detection(const cv::Mat& image) override;    

        ~TRTInfer()
        {
            for (void* buffer : buffers_)
            {
                cudaFree(buffer);
            }
        }


};