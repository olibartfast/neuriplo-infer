#include "CommandLineParser.hpp"
#include <iostream>
#include <glog/logging.h>
#include "utils.hpp"

const std::string CommandLineParser::params = 
    "{ help h   |   | print help message }"
    "{ type     |  yolov10 | yolov4, yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolo11, rtdetr, rtdetrul, dfine}"
    "{ source s   | <none>  | path to image or video source}"
    "{ labels lb  |<none>  | path to class labels}"
    "{ weights w  | <none>  | path to models weights}"
    "{ use-gpu   | false  | activate gpu support}"
    "{ min_confidence | 0.25   | optional min confidence}"
    "{ batch b | 1 | Batch size}"
    "{ input_sizes is | | Input sizes for each model input. Format: CHW;CHW;... (e.g., '3,224,224' for single input or '3,224,224;3,224,224' for two inputs, '3,640,640;2' for rtdetr/dfine models) }"
    "{ warmup     | false  | enable GPU warmup}"
    "{ benchmark  | false  | enable benchmarking}"
    "{ iterations | 10     | number of iterations for benchmarking}";

AppConfig CommandLineParser::parseCommandLineArguments(int argc, char *argv[]) {
    cv::CommandLineParser parser(argc, argv, params);
    parser.about("Detect objects from video or image input source");

    if (parser.has("help")) {
        printHelpMessage(parser);
        std::exit(1);
    }

    validateArguments(parser);

    AppConfig config;
    config.source = parser.get<std::string>("source");
    config.use_gpu = parser.get<bool>("use-gpu");
    config.enable_warmup = parser.get<bool>("warmup");
    config.enable_benchmark = parser.get<bool>("benchmark");
    config.benchmark_iterations = parser.get<int>("iterations");
    config.confidenceThreshold = parser.get<float>("min_confidence");
    config.detectorType = parser.get<std::string>("type");
    config.weights = parser.get<std::string>("weights");
    config.labelsPath = parser.get<std::string>("labels");
    config.batch_size = parser.get<int>("batch");

    std::vector<std::vector<int64_t>> input_sizes;
    if(parser.has("input_sizes")) {
        LOG(INFO) << "Parsing input sizes..." << parser.get<std::string>("input_sizes") << std::endl;
        input_sizes = parseInputSizes(parser.get<std::string>("input_sizes"));
        // Output the parsed sizes
        LOG(INFO) << "Parsed input sizes:\n";
        for (const auto& size : input_sizes) {
            LOG(INFO) << "(";
            for (size_t i = 0; i < size.size(); ++i) {
                LOG(INFO) << size[i];
                if (i < size.size() - 1) {
                    LOG(INFO) << ",";
                }
            }
            LOG(INFO)<< ")\n";
        }               
    }
    else {
        LOG(INFO) << "No input sizes provided. Will use default model configuration." << std::endl;
    }    
    // copy input sizes to config
    config.input_sizes = input_sizes;

    return config;
}

void CommandLineParser::printHelpMessage(const cv::CommandLineParser& parser) {
    parser.printMessage();
}

void CommandLineParser::validateArguments(const cv::CommandLineParser& parser) {
    if (!parser.check()) {
        parser.printErrors();
        std::exit(1);
    }

    std::string source = parser.get<std::string>("source");
    if (source.empty()) {
        LOG(ERROR) << "Cannot open video stream";
        std::exit(1);
    }

    std::string weights = parser.get<std::string>("weights");
    if (!isFile(weights)) {
        LOG(ERROR) << "Weights file " << weights << " doesn't exist";
        std::exit(1);
    }

    std::string labelsPath = parser.get<std::string>("labels");
    if (!isFile(labelsPath)) {
        LOG(ERROR) << "Labels file " << labelsPath << " doesn't exist";
        std::exit(1);
    }
}