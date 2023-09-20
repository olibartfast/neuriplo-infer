#include "YoloVn.hpp"

YoloVn::YoloVn(const std::string& model_path, bool use_gpu,
               float confidenceThreshold, size_t network_width,
               size_t network_height)
    : Detector{model_path, use_gpu, confidenceThreshold,
           network_width, network_height}
{
    logger_->info("Initializing Libtorch from model {}", model_path);
    if (use_gpu && torch::cuda::is_available())
    {
        device_ = torch::kCUDA;
        logger_->info("Using CUDA GPU");
    }
    else
    {
        device_ = torch::kCPU;
        logger_->info("Using CPU");
    }

    module_ = torch::jit::load(model_path, device_);

    // In libtorch i didn't find a way to get the input layer
    network_width_ = 640;
    network_height_ = 640;
    channels_ = 3;
}


cv::Rect YoloVn::get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
{
    float r_w = network_width_ / static_cast<float>(imgSz.width);
    float r_h = network_height_ / static_cast<float>(imgSz.height);
    
    int l, r, t, b;
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (network_height_ - r_w * imgSz.height) / 2;
        b = bbox[1] + bbox[3] / 2.f - (network_height_ - r_w * imgSz.height) / 2;
        l /= r_w;
        r /= r_w;
        t /= r_w;
        b /= r_w;
    }
    else {
        l = bbox[0] - bbox[2] / 2.f - (network_width_ - r_h * imgSz.width) / 2;
        r = bbox[0] + bbox[2] / 2.f - (network_width_ - r_h * imgSz.width) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l /= r_h;
        r /= r_h;
        t /= r_h;
        b /= r_h;
}

    // Clamp the coordinates within the image bounds
    l = std::max(0, std::min(l, imgSz.width - 1));
    r = std::max(0, std::min(r, imgSz.width - 1));
    t = std::max(0, std::min(t, imgSz.height - 1));
    b = std::max(0, std::min(b, imgSz.height - 1));

    return cv::Rect(l, t, r - l, b - t);
}


std::vector<Detection> YoloVn::run_detection(const cv::Mat& image)
{
    // Preprocess the input image
    std::vector<float> input_tensor = preprocess_image(image);

    // Convert the input tensor to a Torch tensor
    torch::Tensor input = torch::from_blob(input_tensor.data(), { 1, channels_, network_height_, network_width_ }, torch::kFloat32);
    input = input.to(device_);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto output = module_.forward(inputs);
    torch::Tensor output_tensor0 = output.toTuple()->elements()[0].toTensor();

    // Convert the output tensors to CPU and extract data
    output_tensor0 = output_tensor0.to(torch::kCPU).contiguous();
    const float* output0 = output_tensor0.data_ptr<float>();

    // Get the shapes of the output tensors
    std::vector<int64_t> shape0 = output_tensor0.sizes().vec();
    cv::Size frame_size(image.cols, image.rows);

    // Perform post-processing on the output and return the detections
    return postprocess(output0, shape0, frame_size);
}




std::vector<float> YoloVn::preprocess_image(const cv::Mat &image)
{
    cv::Mat blob;
    cv::cvtColor(image, blob, cv::COLOR_BGR2RGB);
    int target_width, target_height, offset_x, offset_y;
    float resize_ratio_width = static_cast<float>(network_width_) / static_cast<float>(image.cols);
    float resize_ratio_height = static_cast<float>(network_height_) / static_cast<float>(image.rows);

    if (resize_ratio_height > resize_ratio_width)
    {
        target_width = network_width_;
        target_height = resize_ratio_width * image.rows;
        offset_x = 0;
        offset_y = (network_height_ - target_height) / 2;
    }
    else
    {
        target_width = resize_ratio_height * image.cols;
        target_height = network_height_;
        offset_x = (network_width_ - target_width) / 2;
        offset_y = 0;
    }

    cv::Mat resized_image(target_height, target_width, CV_8UC3);
    cv::resize(blob, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat output_image(network_width_, network_height_, CV_8UC3, cv::Scalar(128, 128, 128));
    resized_image.copyTo(output_image(cv::Rect(offset_x, offset_y, resized_image.cols, resized_image.rows)));   
    output_image.convertTo(output_image, CV_32FC3, 1.f / 255.f);        

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

std::vector<Detection> YoloVn::postprocess(const float*  output0, const  std::vector<int64_t>& shape0,  const cv::Size& frame_size)
{

    const auto offset = 5;
    const auto num_classes = shape0[2] - offset; // 1 x 25200 x 85

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    
    std::vector<std::vector<float>> picked_proposals;

    // Get all the YOLO proposals
    for (int i = 0; i < shape0[1]; ++i) {
        if(output0[4] > confidenceThreshold_)
        {
            const float* scoresPtr = output0 + 5;
            auto maxSPtr = std::max_element(scoresPtr, scoresPtr + num_classes);
            float score = *maxSPtr;
            if (score > confidenceThreshold_) {
                boxes.emplace_back(get_rect(frame_size, std::vector<float>(output0, output0 + 4)));
                int label = maxSPtr - scoresPtr;
                confs.emplace_back(score);
                classIds.emplace_back(label);
            }

        }
        output0 += shape0[2]; 
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confidenceThreshold_, nms_threshold_, indices);
    std::vector<Detection> detections;
    for (int i = 0; i < indices.size(); i++)
    {
        Detection det;
        int idx = indices[i];
        det.label = classIds[idx];
        det.bbox = boxes[idx];
        det.score = confs[idx];
        detections.emplace_back(det);
    }
    return detections;
}
