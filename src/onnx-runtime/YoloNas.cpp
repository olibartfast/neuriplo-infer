#include "YoloNas.hpp"

YoloNas::YoloNas(const std::string& model_path, bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    OnnxRuntimeInference{model_path, use_gpu, confidenceThreshold,
            network_width,
            network_height}
{
    logger_->info("Running YoloNas onnx runtime");
  
}


std::vector<Detection> YoloNas::run_detection(const cv::Mat& image)
{
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

    const float* output0 = output_ort_tensors[0].GetTensorData<float>();

    const auto& shape0_ref = output_ort_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        assert(output_ort_tensors.size() == output_names_.size());

    const float* output1 = output_ort_tensors[1].GetTensorData<float>();

    const auto& shape1_ref = output_ort_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    std::vector<int64_t> shape0(shape0_ref.begin(), shape0_ref.end()); // boxes 1x8400x4
    std::vector<int64_t> shape1(shape1_ref.begin(), shape1_ref.end()); // scores 1x8400x80
    cv::Size frame_size(image.cols, image.rows);
    return postprocess(output0, output1, shape0, shape1, frame_size);   
}


std::vector<float> YoloNas::preprocess_image(const cv::Mat& image)
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



std::vector<Detection> YoloNas::postprocess(const float* output0, const float* output1 ,const std::vector<int64_t>& shape0, const std::vector<int64_t>& shape1, const cv::Size& frame_size)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // idx 0 boxes, idx 1 scores
    int rows = shape0[1]; // 8400
    int dimensions_boxes = shape0[2];  // 4
    int dimensions_scores = shape1[2]; // num classes (80)

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        auto maxSPtr = std::max_element(output1, output1 + dimensions_scores);
        float score = *maxSPtr;
        if (score >= confidenceThreshold_) 
        {
            int label = maxSPtr - output1;
            confidences.push_back(score);
            classIds.push_back(label);
            float r_w = (frame_size.width * 1.0) / network_width_;
            float r_h = (frame_size.height * 1.0) / network_height_ ;
            std::vector<float> bbox(&output0[0], &output0[4]);

            int left = (int)(bbox[0] * r_w);
            int top = (int)(bbox[1] * r_h);
            int width = (int)((bbox[2] - bbox[0]) * r_w);
            int height = (int)((bbox[3] - bbox[1]) * r_h);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        // Jump to the next column.
        output1 += dimensions_scores;
        output0 += dimensions_boxes;
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold_, nms_threshold_, indices);
    std::vector<Detection> detections;
    for (int i = 0; i < indices.size(); i++) 
    {
        Detection det;
        int idx = indices[i];
        det.label = classIds[idx];
        det.bbox = boxes[idx];
        det.score = confidences[idx];
        detections.emplace_back(det);
    }
    return detections; 
}