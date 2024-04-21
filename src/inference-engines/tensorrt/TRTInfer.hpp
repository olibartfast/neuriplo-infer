#pragma once
#include "InferenceInterface.hpp"
#include <NvInfer.h>  // for TensorRT API
#include <cuda_runtime_api.h>  // for CUDA runtime API
#include <fstream>

#include "Logger.hpp"

class TRTInfer : public InferenceInterface
{
    protected:
        std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
        nvinfer1::IExecutionContext* context_{nullptr};
        std::vector<void*> buffers_;
        nvinfer1::IRuntime* runtime_{nullptr};
        size_t num_inputs_{0};
        size_t num_outputs_{0};

    public:
        TRTInfer(const std::string& model_path);

        // Create execution context and allocate input/output buffers
        void createContextAndAllocateBuffers();

        void initializeBuffers(const std::string& engine_path);

        // calculate size of tensor
        size_t getSizeByDim(const nvinfer1::Dims& dims);    

        void infer();

        std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;

        ~TRTInfer()
        {
            for (void* buffer : buffers_)
            {
                cudaFree(buffer);
            }
        }


};