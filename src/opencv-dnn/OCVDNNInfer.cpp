#include "OCVDNNInfer.hpp"

OCVDNNInfer::OCVDNNInfer(const std::string& modelConfiguration,
    const std::string& modelBinary,
    bool use_gpu,
    float confidenceThreshold,
    size_t network_width,
    size_t network_height) : 
    Detector{modelBinary, use_gpu, confidenceThreshold,
            network_width,
            network_height}
{

        logger_->info("Running {} using OpenCV DNN runtime", modelBinary);
        net_ = modelConfiguration.empty() ? cv::dnn::readNet(modelBinary) : cv::dnn::readNetFromDarknet(modelConfiguration, modelBinary);
        if (net_.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "weights-file: " << modelBinary << std::endl;
            exit(-1);
        }
        outLayers_ = net_.getUnconnectedOutLayers();
        outLayerType_ = net_.getLayer(outLayers_[0])->type;
        outNames_ = net_.getUnconnectedOutLayersNames();


}

cv::Rect OCVDNNInfer::get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
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


std::vector<Detection> OCVDNNInfer::run_detection(const cv::Mat& image) 
{
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;

    cv::Mat inputBlob = preprocess_image(image);
    std::vector<cv::Mat> outs;
	net_.setInput(inputBlob);
    net_.forward(outs, outNames_);
    
    for (const auto& output : outs) {
        // Extracting dimensions of the output tensor
        std::vector<int64_t> shape;
        for (int i = 0; i < output.dims; ++i) {
            shape.push_back(output.size[i]);
        }
        shapes.push_back(shape);

        // Extracting data and converting to float
        const float* data = output.ptr<float>();
        outputs.emplace_back(data, data + output.total());
    }
    cv::Size frame_size(image.cols, image.rows);
    return postprocess(outputs, shapes, frame_size);
}
