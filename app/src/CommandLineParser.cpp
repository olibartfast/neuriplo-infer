#include "CommandLineParser.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <glog/logging.h>
#include "utils.hpp"

const std::string CommandLineParser::params =
    "{ help h   |   | print help message }"
    "{ type     |  yolov10 | Object Detection: yolo, yolov4, yolov7e2e, yolov10, yolonas, rtdetr, rtdetrul, rfdetr, owlv2 | Classification: torchvisionclassifier, tensorflowclassifier, vitclassifier, timesformer | Instance Segmentation: yoloseg | Optical Flow: raft | Pose Estimation: vitpose }"
    "{ source s   | <none>  | path to image or video source}"
    "{ labels lb  |<none>  | path to class labels}"
    "{ text_prompts tp | | semicolon-separated text prompts for open-vocabulary detection (e.g. 'cat;dog;bus')}"
    "{ prompt | | freeform prompt for multimodal understanding models }"
    "{ output_format | | optional multimodal output hint (text or json) }"
    "{ sample_stride | 0 | optional uniform frame sampling stride for multimodal video tasks }"
    "{ max_frames | 0 | optional cap on sampled frames for multimodal video tasks }"
    "{ tokenizer_vocab | | path to tokenizer vocab.json for open-vocabulary detection }"
    "{ tokenizer_merges | | path to tokenizer merges.txt for open-vocabulary detection }"
    "{ weights w  | <none>  | path to models weights}"
    "{ use-gpu   | false  | activate gpu support}"
    "{ min_confidence | 0.25   | optional min confidence}"
    "{ nms_threshold  | 0.45   | NMS IoU threshold (YOLO-based detectors/segmenters) }"
    "{ mask_threshold | 0.50   | Mask binarization threshold (instance segmentation) }"
    "{ batch b | 1 | Batch size}"
    "{ input_sizes is | | Input sizes for each model input. Format: CHW;CHW;... (e.g., '3,224,224' for single input or '3,224,224;3,224,224' for two inputs, '3,640,640;2' for rtdetr/dfine models) }"
    "{ warmup     | false  | enable GPU warmup}"
    "{ benchmark  | false  | enable benchmarking}"
    "{ iterations | 10     | number of iterations for benchmarking}"
    "{ num_frames nf | 0   | number of frames for video classification (0 = use model default, e.g., 16 for VideoMAE)}";

AppConfig CommandLineParser::parseCommandLineArguments(int argc, char *argv[]) {
    cv::CommandLineParser parser(argc, argv, params);
    parser.about("Detect objects from video or image input source");

    if (parser.has("help")) {
        printHelpMessage(parser);
        std::exit(1);
    }

    validateArguments(parser);

    AppConfig config;
    std::string source_str = parser.get<std::string>("source");
    config.sources = split(source_str, ',');
    config.use_gpu = parser.get<bool>("use-gpu");
    config.enable_warmup = parser.get<bool>("warmup");
    config.enable_benchmark = parser.get<bool>("benchmark");
    config.benchmark_iterations = parser.get<int>("iterations");
    config.confidenceThreshold = parser.get<float>("min_confidence");
    config.nmsThreshold = parser.get<float>("nms_threshold");
    config.maskThreshold = parser.get<float>("mask_threshold");
    config.detectorType = parser.get<std::string>("type");
    config.weights = parser.get<std::string>("weights");
    config.labelsPath = parser.get<std::string>("labels");
    config.tokenizerVocabPath = parser.get<std::string>("tokenizer_vocab");
    config.tokenizerMergesPath = parser.get<std::string>("tokenizer_merges");
    config.batch_size = parser.get<int>("batch");
    {
        const std::string prompts = parser.get<std::string>("text_prompts");
        if (!prompts.empty()) {
            config.textPrompts = split(prompts, ';');
        }
    }
    {
        const std::string prompt = parser.get<std::string>("prompt");
        if (!prompt.empty()) {
            config.taskExtraParams["prompt"] = prompt;
        }
    }
    {
        std::string outputFormat = parser.get<std::string>("output_format");
        if (!outputFormat.empty()) {
            std::transform(outputFormat.begin(), outputFormat.end(), outputFormat.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            config.taskExtraParams["output_format"] = outputFormat;
        }
    }
    {
        const int sampleStride = parser.get<int>("sample_stride");
        if (sampleStride > 0) {
            config.taskExtraParams["sample_stride"] = std::to_string(sampleStride);
        }
    }
    {
        const int maxFrames = parser.get<int>("max_frames");
        if (maxFrames > 0) {
            config.taskExtraParams["max_frames"] = std::to_string(maxFrames);
        }
    }

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

    // Parse num_frames for video classification
    config.num_frames = parser.get<int>("num_frames");
    if (config.num_frames > 0) {
        LOG(INFO) << "Using " << config.num_frames << " frames for video classification";
    }

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
    
    // Validate each source file exists
    std::vector<std::string> sources = split(source, ',');
    for (const auto& src : sources) {
        if (!isFile(src) && !isDirectory(src)) {
            LOG(ERROR) << "Source file/directory " << src << " doesn't exist";
            std::exit(1);
        }
    }

    std::string weights = parser.get<std::string>("weights");
    if (!isFile(weights)) {
        LOG(ERROR) << "Weights file " << weights << " doesn't exist";
        std::exit(1);
    }

    std::string labelsPath = parser.get<std::string>("labels");
    if (!labelsPath.empty() && !isFile(labelsPath)) {
        LOG(ERROR) << "Labels file " << labelsPath << " doesn't exist";
        std::exit(1);
    }

    const std::string detectorType = parser.get<std::string>("type");
    const std::string normalizedType = normalizeModelType(detectorType);
    std::string outputFormat = parser.get<std::string>("output_format");
    const int sampleStride = parser.get<int>("sample_stride");
    const int maxFrames = parser.get<int>("max_frames");

    std::transform(outputFormat.begin(), outputFormat.end(), outputFormat.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (normalizedType == "owlv2" || normalizedType == "owlvit") {
        const std::string textPrompts = parser.get<std::string>("text_prompts");
        const std::string tokenizerVocab = parser.get<std::string>("tokenizer_vocab");
        const std::string tokenizerMerges = parser.get<std::string>("tokenizer_merges");

        if (textPrompts.empty()) {
            LOG(ERROR) << "Open-vocabulary detection requires --text_prompts";
            std::exit(1);
        }
        if (tokenizerVocab.empty()) {
            LOG(ERROR) << "Open-vocabulary detection requires --tokenizer_vocab";
            std::exit(1);
        }
        if (!isFile(tokenizerVocab)) {
            LOG(ERROR) << "Tokenizer vocab file " << tokenizerVocab << " doesn't exist";
            std::exit(1);
        }
        if (tokenizerMerges.empty()) {
            LOG(ERROR) << "Open-vocabulary detection requires --tokenizer_merges";
            std::exit(1);
        }
        if (!isFile(tokenizerMerges)) {
            LOG(ERROR) << "Tokenizer merges file " << tokenizerMerges << " doesn't exist";
            std::exit(1);
        }
    }

    if (!outputFormat.empty() && outputFormat != "text" && outputFormat != "json") {
        LOG(ERROR) << "--output_format must be either 'text' or 'json'";
        std::exit(1);
    }

    if (sampleStride < 0) {
        LOG(ERROR) << "--sample_stride must be >= 0";
        std::exit(1);
    }

    if (maxFrames < 0) {
        LOG(ERROR) << "--max_frames must be >= 0";
        std::exit(1);
    }
}
