#include "CommandLineParser.hpp"
#include <iostream>
#include <glog/logging.h>
#include "utils.hpp"

const std::string CommandLineParser::params = 
    "{ help h   |   | print help message }"
    "{ type     |  yolov10 | yolov4, yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolo11, rtdetr, rtdetrul, dfine}"
    "{ source s   | <none>  | path to image or video source}"
    "{ labels lb  |<none>  | path to class labels}"
    "{ config c   | <none>  | optional model configuration file}"
    "{ weights w  | <none>  | path to models weights}"
    "{ use-gpu   | false  | activate gpu support}"
    "{ min_confidence | 0.25   | optional min confidence}"
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
    config.config = parser.get<std::string>("config");
    config.weights = parser.get<std::string>("weights");
    config.labelsPath = parser.get<std::string>("labels");

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

    std::string config = parser.get<std::string>("config");
    if (!config.empty() && !isFile(config)) {
        LOG(ERROR) << "Conf file " << config << " doesn't exist";
        std::exit(1);
    }

    std::string weights = parser.get<std::string>("weights");
    if (!isFile(weights) && getFileExtension(config) != "xml") {
        LOG(ERROR) << "Weights file " << weights << " doesn't exist";
        std::exit(1);
    }

    std::string labelsPath = parser.get<std::string>("labels");
    if (!isFile(labelsPath)) {
        LOG(ERROR) << "Labels file " << labelsPath << " doesn't exist";
        std::exit(1);
    }
}