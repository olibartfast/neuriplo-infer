#include "ORTInfer.hpp"

ORTInfer::ORTInfer(const std::string& model_path, bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{model_path, use_gpu, confidenceThreshold,
            network_width,
            network_height}
{
    env_=Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Onnx Runtime Inference");

    Ort::SessionOptions session_options;

    if (use_gpu)
    {
        // Check if CUDA GPU is available
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        logger_->info("Available providers:");
        bool is_found = false;
        for (const auto& p : providers)
        {
            logger_->info("{}", p);
            if (p.find("CUDA") != std::string::npos)
            {
                // CUDA GPU is available, use it
                logger_->info("Using CUDA GPU");
                OrtCUDAProviderOptions cuda_options;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                is_found = true;
                break;
            }
        }
        if (!is_found)
        {
            // CUDA GPU is not available, fall back to CPU
            logger_->info("CUDA GPU not available, falling back to CPU");
            session_options = Ort::SessionOptions();
        }
    }
    else
    {
        logger_->info("Using CPU");
        session_options = Ort::SessionOptions();
    }

    try
    {
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
    }
    catch (const Ort::Exception& ex)
    {
        logger_->error("Failed to load the ONNX model: {}", ex.what());
        std::exit(1);
    }

    Ort::AllocatorWithDefaultOptions allocator;
    logger_->info("Input Node Name/Shape ({}):", session_.GetInputCount());
    for (std::size_t i = 0; i < session_.GetInputCount(); i++)
    {
        input_names_.emplace_back(session_.GetInputNameAllocated(i, allocator).get());
        const auto input_shapes = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        logger_->info("\t{} : {}", input_names_.at(i), print_shape(input_shapes));
        input_shapes_.emplace_back(input_shapes);
    }
    network_width_ = static_cast<int>(input_shapes_[0][3]);
    network_height_ = static_cast<int>(input_shapes_[0][2]);
    channels_ = static_cast<int>(input_shapes_[0][1]);
    logger_->info("channels {}", channels_);
    logger_->info("width {}", network_width_);
    logger_->info("height {}", network_height_);

    // print name/shape of outputs
    logger_->info("Output Node Name/Shape ({}):", session_.GetOutputCount());
    for (std::size_t i = 0; i < session_.GetOutputCount(); i++)
    {
        output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
        auto output_shapes = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        logger_->info("\t{} : {}", output_names_.at(i), print_shape(output_shapes));
        output_shapes_.emplace_back(output_shapes);
    }
}

std::string ORTInfer::print_shape(const std::vector<std::int64_t>& v)
{
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}
std::vector<Detection> ORTInfer::run_detection(const cv::Mat& image)
{
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<float>> input_tensors(session_.GetInputCount());
    std::vector<Ort::Value> in_ort_tensors;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    for (size_t i = 0; i < session_.GetInputCount(); ++i)
    {
        input_tensors[i] = preprocess_image(image);
        in_ort_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensors[i].data(),
            input_tensors[i].size(),
            input_shapes_[i].data(),
            input_shapes_[i].size()
        ));
    }

    // Run inference
    std::vector<const char*> input_names_char(input_names_.size());
    std::transform(input_names_.begin(), input_names_.end(), input_names_char.begin(),
        [](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names_.size());
    std::transform(output_names_.begin(), output_names_.end(), output_names_char.begin(),
        [](const std::string& str) { return str.c_str(); });

    std::vector<Ort::Value> output_ort_tensors = session_.Run(
        Ort::RunOptions{ nullptr },
        input_names_char.data(),
        in_ort_tensors.data(),
        in_ort_tensors.size(),
        output_names_char.data(),
        output_names_.size()
    );

    // Process output tensors
    assert(output_ort_tensors.size() == output_names_.size());

    for (const Ort::Value& output_tensor : output_ort_tensors)
    {
        const float* output_data = output_tensor.GetTensorData<float>();
        const auto& shape_ref = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> shape(shape_ref.begin(), shape_ref.end());

        size_t num_elements = 1;
        for (int64_t dim : shape) {
            num_elements *= dim;
        }
        
        outputs.emplace_back(std::vector<float>(output_data, output_data + num_elements));
        shapes.emplace_back(shape);
    }

    // Assuming postprocess function takes outputs and shapes as arguments
    cv::Size frame_size(image.cols, image.rows);
    return postprocess(outputs, shapes, frame_size);
}
