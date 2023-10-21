#include "TRTInfer.hpp"

TRTInfer::TRTInfer(const std::string& model_path, bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{model_path, use_gpu, confidenceThreshold,
            network_width,
            network_height}
{

}

void TRTInfer::initializeBuffers(const std::string& engine_path)
{


    // Create TensorRT runtime and deserialize engine
    Logger logger;
    // Create TensorRT runtime
    runtime_ = nvinfer1::createInferRuntime(logger);

    // Load engine file
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file)
    {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    engine_file.seekg(0, std::ios::end);
    size_t file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(file_size);
    engine_file.read(engine_data.data(), file_size);
    engine_file.close();

    // Deserialize engine
     engine_.reset(
        runtime_->deserializeCudaEngine(engine_data.data(), file_size),
        [](nvinfer1::ICudaEngine* engine) { engine->destroy(); });



    // Create execution context and allocate input/output buffers
    createContextAndAllocateBuffers();
}

// calculate size of tensor
size_t TRTInfer::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}



void TRTInfer::createContextAndAllocateBuffers()
{
    context_ = engine_->createExecutionContext();
    buffers_.resize(engine_->getNbBindings());
    std::vector<nvinfer1::Dims> output_dims;
    for (int i = 0; i < engine_->getNbBindings(); ++i)
    {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        auto binding_size = getSizeByDim(engine_->getBindingDimensions(i)) * sizeof(float);
        cudaMalloc(&buffers_[i], binding_size);
        if (engine_->bindingIsInput(i))
        {
            const auto input_shape = std::vector{ dims.d[0], dims.d[1], dims.d[2], dims.d[3] };
            network_width_ = dims.d[3];
            network_height_ = dims.d[2];
            channels_ = dims.d[1];
        }
        else
        {
            output_dims.emplace_back(engine_->getBindingDimensions(i));
            auto size = getSizeByDim(dims);
            h_outputs_.emplace_back(std::vector<float>(size));

            const int* shape_boxes_ptr = reinterpret_cast<const int*>(output_dims[i].d);
            std::vector<int64_t> shape(shape_boxes_ptr, shape_boxes_ptr + output_dims[i].nbDims);
            output_shapes_.emplace_back(shape);
        }
    }
}


void TRTInfer::infer()
{
    if(!context_->enqueueV2(buffers_.data(), 0, nullptr))
    {
        logger_->error("Forward Error !");
        std::exit(1);
    }
        

    for (size_t i = 0; i < h_outputs_.size(); i++)
    {
        cudaMemcpy(h_outputs_[i].data(), buffers_[i + 1], h_outputs_[i].size() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
}
