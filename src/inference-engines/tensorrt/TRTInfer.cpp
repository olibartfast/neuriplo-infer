#include "TRTInfer.hpp"

TRTInfer::TRTInfer(const std::string& model_path) : InferenceInterface{model_path, "", true}
{

    logger_->info("Initializing TensorRT for model {}", model_path);
    initializeBuffers(model_path);
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
        if(dims.d[i] == -1 || dims.d[i] == 0)
        {
            continue;
        }
        size *= dims.d[i];
    }
    return size;
}



void TRTInfer::createContextAndAllocateBuffers()
{
    nvinfer1::Dims profile_dims = engine_->getProfileDimensions(0, 0 /* max batch size index */, nvinfer1::OptProfileSelector::kMIN);
    int max_batch_size = profile_dims.d[0];
    context_ = engine_->createExecutionContext();
    context_->setBindingDimensions(0, profile_dims);
    buffers_.resize(engine_->getNbBindings());
    for (int i = 0; i < engine_->getNbBindings(); ++i)
    {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        auto binding_size = getSizeByDim(dims) * sizeof(float);
        cudaMalloc(&buffers_[i], binding_size);
        if (engine_->bindingIsInput(i))
            continue;
        auto size = getSizeByDim(dims);
        h_outputs_.emplace_back(std::vector<float>(size));
        const int64_t curr_batch = dims.d[0] == -1 ? 1 : dims.d[0];
        const auto out_shape = std::vector<int64_t>{curr_batch, dims.d[1], dims.d[2], dims.d[3] };
        output_shapes_.emplace_back(out_shape);
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

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> TRTInfer::get_infer_results(const cv::Mat& input_blob) 
{

    cudaMemcpy(buffers_[0], input_blob.data, sizeof(float)* get_blob_size(input_blob), cudaMemcpyHostToDevice);
    infer();
    return std::make_tuple(h_outputs_, output_shapes_);
}  
