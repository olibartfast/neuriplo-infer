#include "OCVDNNInfer.hpp"

OCVDNNInfer::OCVDNNInfer(const std::string& modelConfiguration,
    const std::string& modelBinary) : InferenceEngine{modelConfiguration, modelBinary} 
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


std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> OCVDNNInfer::get_infer_results(const cv::Mat& input_blob)
{

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;


    std::vector<cv::Mat> outs;
    net_.setInput(input_blob);
    net_.forward(outs, outNames_);
    
    for (const auto& output : outs) 
    {
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
    return std::make_tuple(outputs, shapes);
}