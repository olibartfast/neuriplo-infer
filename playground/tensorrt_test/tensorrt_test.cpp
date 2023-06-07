#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <numeric>
#include "opencv2/opencv.hpp"

using namespace nvinfer1;

// utilities ----------------------------------------------------------------------------------------------------------
// class to log errors, warnings, and other information during the build and inference phases
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
        }
    }
} gLogger;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                     TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    auto MAX_WORKSPACE_SIZE = 1ULL << 30;
    config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // generate TensorRT engine optimized for the target platform
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());

    IHostMemory* modelStream = engine->serialize();
    std::ofstream p("resnet.engine",std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
}

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

// get classes names
std::vector<std::string> getClassesNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}


// preprocessing stage ------------------------------------------------------------------------------------------------
void preprocessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims)
{
    // read input image
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }


    auto input_width = dims.d[2];
    auto input_height = dims.d[3];
    auto channels = dims.d[1];
    auto input_size = cv::Size(input_width, input_height);
    // resize
    cv::Mat resized;
    cv::resize(frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    // normalize
    cv::Mat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // to tensor
    size_t img_byte_size = flt_image.total() * flt_image.elemSize();  // Allocate a buffer to hold all image elements.
    std::vector<float> cpu_input = std::vector<float>(input_width*input_height*channels);
    std::memcpy(cpu_input.data(), flt_image.data, img_byte_size);

    std::vector<cv::Mat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::Mat(input_size, CV_32FC1, &(cpu_input[i * input_width * input_height])));
    }
    cv::split(flt_image, chw);

    cudaMemcpy(gpu_input, cpu_input.data(), img_byte_size, cudaMemcpyHostToDevice);

}


// post-processing stage ----------------------------------------------------------------------------------------------
void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, const std::string& pathToClassesNames)
{
    // get class names
    auto classes = getClassesNames(pathToClassesNames);

    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims));
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(dims));
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "\n";
        ++i;
    }
}

int main(int argc, char* argv[])
{
    std::cout << argc << std::endl;
    if (argc < 4)
    {
        std::cerr << "usage: " << argv[0] << "./resnet50-trt  path_to_onnx_model/resnet50.onnx path_to_image/turkish_coffee.jpg path_to_classes_names/imagenet_classes.txt \n";
        return -1;
    }
    std::string model_path(argv[1]);
    std::string image_path(argv[2]);
    std::string path_to_classes_names(argv[3]);

    std::cout << model_path << std::endl;
    std::cout << image_path << std::endl;
    std::cout << path_to_classes_names << std::endl;

    // initialize TensorRT engine and parse ONNX model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    parseOnnxModel(model_path, engine, context);

        // get sizes of input and output and allocate memory required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output
    std::vector<void*> buffers(engine->getNbBindings()); // buffers for input and output data
     for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }


    // preprocess input data
    preprocessImage(image_path, (float *) buffers[0], input_dims[0]);

    // inference
    std::cout << "Inference"<< std::endl;
    
    if(context->enqueueV2(buffers.data(), 0, nullptr))
        std::cout << "Forward success !" << std::endl;
    else
        std::cout << "Forward Error !" << std::endl;
    
    // postprocess results
    std::cout << "Postprocess"<< std::endl;
    postprocessResults((float *) buffers[1], output_dims[0], path_to_classes_names);

    for (void* buf : buffers)
    {
        cudaFree(buf);
    }       

    std::cout << "Finished"<< std::endl;

    return 0;
}

