#include "OCVDNNInfer.hpp"

OCVDNNInfer::OCVDNNInfer(const std::string& weights, const std::string& modelConfiguration) : InferenceInterface{weights, modelConfiguration} 
{

        logger_->info("Running {} using OpenCV DNN runtime", weights);
        net_ = modelConfiguration.empty() ? cv::dnn::readNet(weights) : cv::dnn::readNetFromDarknet(modelConfiguration, weights);
        if (net_.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "weights-file: " << weights << std::endl;
            exit(-1);
        }
        outLayers_ = net_.getUnconnectedOutLayers();
        outLayerType_ = net_.getLayer(outLayers_[0])->type;
        outNames_ = net_.getUnconnectedOutLayersNames();


}


std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> OCVDNNInfer::get_infer_results(const cv::Mat& input_blob)
{

        std::vector<std::vector<std::any>> outputs;
        std::vector<std::vector<int64_t>> shapes;
        std::vector<std::string> layerTypes;

        std::vector<cv::Mat> outs;
        net_.setInput(input_blob);
        net_.forward(outs, outNames_);

        for (size_t i = 0; i < outs.size(); ++i) {
            const auto& output = outs[i];
            // Extracting dimensions of the output tensor
            std::vector<int64_t> shape;
            for (int j = 0; j < output.dims; ++j) {
                shape.push_back(output.size[j]);
            }
            shapes.push_back(shape);

            // Extracting data
            if (output.type() == CV_32F) {
                const float* data = output.ptr<float>();
                outputs.emplace_back(data, data + output.total());
            } 
            else if (output.type() == CV_64F) {
                const int64_t* data = output.ptr<int64_t>();
                outputs.emplace_back(data, data + output.total());
            } 
            else {
                std::cerr << "Unsupported data type\n";
            }
        }

        return std::make_tuple(outputs, shapes);
}