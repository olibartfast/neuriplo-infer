#include "RtDetr.hpp"


RtDetr::RtDetr(const std::string& model_path, bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{model_path, use_gpu, confidenceThreshold,
            network_width,
            network_height}
{
    logger_->info("Initializing RT-DETR TensorRT");
    initializeBuffers(model_path);
}


// Create the TensorRT runtime and deserialize the engine
std::shared_ptr<nvinfer1::ICudaEngine> RtDetr::createRuntimeAndDeserializeEngine(const std::string& engine_path, Logger& logger, nvinfer1::IRuntime*& runtime)
{
    // Create TensorRT runtime
    runtime = nvinfer1::createInferRuntime(logger);

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
    std::shared_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(engine_data.data(), file_size),
        [](nvinfer1::ICudaEngine* engine) { engine->destroy(); });

    return engine;
}


void RtDetr::createContextAndAllocateBuffers(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext*& context, std::vector<void*>& buffers, std::vector<nvinfer1::Dims>& output_dims, std::vector<std::vector<float>>& h_outputs)
{
    context = engine->createExecutionContext();
    buffers.resize(engine->getNbBindings());
    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            const auto input_shape = std::vector{ dims.d[0], dims.d[1], dims.d[2], dims.d[3] };
            network_width_ = dims.d[3];
            network_height_ = dims.d[2];
            channels_ = dims.d[1];
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
            auto size = getSizeByDim(dims);
            h_outputs.emplace_back(std::vector<float>(size));
        }
    }
}


std::vector<float> RtDetr::preprocess_image(const cv::Mat& image)
{
    cv::Mat blob;
    cv::cvtColor(image, blob, cv::COLOR_BGR2RGB);
    cv::Mat resized_image(network_height_, network_width_, CV_8UC3);
    cv::resize(blob, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat output_image;
    resized_image.convertTo(output_image, CV_32FC3, 1.f / 255.f);        

    size_t img_byte_size = output_image.total() * output_image.elemSize();  // Allocate a buffer to hold all image elements.
    std::vector<float> input_data = std::vector<float>(network_width_ * network_height_ * channels_);
    std::memcpy(input_data.data(), output_image.data, img_byte_size);

    std::vector<cv::Mat> chw;
    for (size_t i = 0; i < channels_; ++i)
    {
        chw.emplace_back(cv::Mat(cv::Size(network_width_, network_height_), CV_32FC1, &(input_data[i * network_width_ * network_height_])));
    }
    cv::split(output_image, chw);

    return input_data;    
}



std::vector<Detection> RtDetr::run_detection(const cv::Mat& image)
{
    // Preprocess the input image
    std::vector<float> h_input_data = preprocess_image(image);
    cudaMemcpy(buffers_[0], h_input_data.data(), sizeof(float)*h_input_data.size(), cudaMemcpyHostToDevice);

    if(!context_->enqueueV2(buffers_.data(), 0, nullptr))
    {
        logger_->error("Forward Error !");
        std::exit(1);
    }
        

    for (size_t i = 0; i < h_outputs_.size(); i++)
        cudaMemcpy(h_outputs_[i].data(), buffers_[i + 1], h_outputs_[i].size() * sizeof(float), cudaMemcpyDeviceToHost);

    const float* output_boxes = h_outputs_[0].data();

    const int* shape_boxes_ptr = reinterpret_cast<const int*>(output_dims_[0].d);
    std::vector<int64_t> shape_boxes(shape_boxes_ptr, shape_boxes_ptr + output_dims_[0].nbDims);
    cv::Size frame_size(image.cols, image.rows);
    return postprocess(output_boxes, shape_boxes, frame_size);   
}  



void RtDetr::initializeBuffers(const std::string& engine_path)
{
    // Create logger
    Logger logger;

    // Create TensorRT runtime and deserialize engine
    engine_ = createRuntimeAndDeserializeEngine(engine_path, logger, runtime_);

    // Create execution context and allocate input/output buffers
    createContextAndAllocateBuffers(engine_.get(), context_, buffers_, output_dims_, h_outputs_);
}

// calculate size of tensor
size_t RtDetr::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}


std::vector<Detection> RtDetr::postprocess(const float*  output0, const  std::vector<int64_t>& shape0,  const cv::Size& frame_size)
{

     std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // idx 0 boxes, idx 1 scores
    int rows = shape0[1]; // 300
    int dimensions_scores = shape0[2] - 4; // num classes (80)

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        auto maxSPtr = std::max_element(output0 + 4 , output0 +  4 +dimensions_scores);
        float score = *maxSPtr;
        if (score >= 0.45) 
        {
            int label = maxSPtr - output0 - 4;
            confidences.push_back(score);
            classIds.push_back(label);
            float r_w = frame_size.width;
            float r_h = frame_size.height;
            std::vector<float> bbox(&output0[0], &output0[4]);

            float x1 = bbox[0] -bbox[2] / 2.0f;
            float y1 = bbox[1] - bbox[3] / 2.0f;
            float x2 = bbox[0] + bbox[2] / 2.0f;
            float y2 =bbox[1] + bbox[3] / 2.0f;
            x2 *= r_w;
            y2 *= r_h;
            x1 *= r_w;
            y1 *= r_h;
            boxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
        output0 += shape0[2] ;
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold_, nms_threshold_, indices);
    std::vector<Detection> detections;
    for (int i = 0; i < boxes.size(); i++) 
    {
        Detection det;
        int idx = i;
        det.label = classIds[idx];
        det.bbox = boxes[idx];
        det.score = confidences[idx];
        detections.emplace_back(det);
    }
    return detections; 
}