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
    nvinfer1::Dims profile_dims = engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMIN);
    int max_batch_size = profile_dims.d[0];
    context_ = engine_->createExecutionContext();
    context_->setBindingDimensions(0, profile_dims);
    buffers_.resize(engine_->getNbBindings());
    for (int i = 0; i < engine_->getNbBindings(); ++i)
    {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        auto size = getSizeByDim(dims);
        size_t binding_size;
        switch (engine_->getBindingDataType(i)) 
        {
            case nvinfer1::DataType::kFLOAT:
                binding_size = size * sizeof(float);
                break;
            case nvinfer1::DataType::kINT32:
                binding_size = size * sizeof(int);
                break;
            // Add more cases for other data types if needed
            default:
                // Handle unsupported data types
                std::exit(1);
        }
        cudaMalloc(&buffers_[i], binding_size);
        if (engine_->bindingIsInput(i))
        {
            num_inputs_++;
            continue;
        }
        num_outputs_++;
    }
}


std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> TRTInfer::get_infer_results(const cv::Mat& input_blob) 
{
    for(size_t i = 0; i < num_inputs_; i++)
    {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        auto size = getSizeByDim(dims);
        size_t binding_size;
        switch (engine_->getBindingDataType(i)) 
        {
            case nvinfer1::DataType::kFLOAT:
                binding_size = size * sizeof(float);
                break;
            case nvinfer1::DataType::kINT32:
                binding_size = size * sizeof(int32_t);
                break;
            // Add more cases for other data types if needed
            default:
                // Handle unsupported data types
                std::exit(1);
        }

        switch(i)
        {
            case 0:
                cudaMemcpy(buffers_[0], input_blob.data, binding_size, cudaMemcpyHostToDevice);
                break;
            case 1:
                // in rtdetr lyuwenyu version we have a second input 
                std::vector<int32_t> orig_target_sizes = { static_cast<int32_t>(input_blob.size[2]), static_cast<int32_t>(input_blob.size[3]) };
                cudaMemcpy(buffers_[1], orig_target_sizes.data(), binding_size, cudaMemcpyHostToDevice);
                break;
        }
    }

    if(!context_->enqueueV2(buffers_.data(), 0, nullptr))
    {
        logger_->error("Forward Error !");
        std::exit(1);
    }
    
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::vector<std::any>> outputs;
    for (size_t i = 0; i < num_outputs_; i++)
    {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i + num_inputs_); // i + 1 to account for the input buffer
        auto num_elements = getSizeByDim(dims);
        std::vector<std::any> tensor_data;
        switch (engine_->getBindingDataType(i + num_inputs_))
        {
            case nvinfer1::DataType::kFLOAT:
            {
                std::vector<float> output_data_float(num_elements);
                cudaMemcpy(output_data_float.data(), buffers_[i + num_inputs_],  num_elements * sizeof(float), cudaMemcpyDeviceToHost);
                for (size_t k = 0; k < num_elements; ++k) {
                    tensor_data.emplace_back(output_data_float[k]);
                }
                break;
            }
            case nvinfer1::DataType::kINT32:
            {
                std::vector<int> output_data_int(num_elements);
                cudaMemcpy(output_data_int.data(), buffers_[i + num_inputs_],  num_elements * sizeof(int), cudaMemcpyDeviceToHost);
                for (size_t k = 0; k < num_elements; ++k) {
                    tensor_data.emplace_back(output_data_int[k]);
                }
                break;
            }

            // Add more cases for other data types if needed
            default:
                // Handle unsupported data types
                std::exit(1);
                break;
        }
        outputs.emplace_back(tensor_data);
        const int64_t curr_batch = dims.d[0] == -1 ? 1 : dims.d[0];
        const auto out_shape = std::vector<int64_t>{curr_batch, dims.d[1], dims.d[2], dims.d[3] };
        output_shapes.emplace_back(out_shape);
    } 

    return std::make_tuple(outputs, output_shapes);
}  
