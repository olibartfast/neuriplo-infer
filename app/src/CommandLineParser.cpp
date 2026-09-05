#include "CommandLineParser.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cctype>
#include <glog/logging.h>
#include <iostream>

const std::string CommandLineParser::params =
    "{ help h   |   | print help message }"
    "{ capabilities | false | print machine-readable capabilities JSON and "
    "exit }"
    "{ type     |  yolov10 | Object Detection: yolo, yolov4, yolov7e2e, "
    "yolov10, yolonas, rtdetr, rtdetrul, rfdetr, owlv2 | Classification: "
    "torchvisionclassifier, tensorflowclassifier, vitclassifier, timesformer | "
    "Instance Segmentation: yoloseg | Optical Flow: raft | Pose Estimation: "
    "vitpose }"
    "{ source s   | <none>  | path to image or video source}"
    "{ labels lb  |<none>  | path to class labels}"
    "{ text_prompts tp | | semicolon-separated text prompts for "
    "open-vocabulary detection (e.g. 'cat;dog;bus')}"
    "{ prompt | | freeform prompt for multimodal understanding models }"
    "{ mmproj | | path to multimodal projector GGUF for VLM image inference }"
    "{ output_format | | optional multimodal output hint (text or json) }"
    "{ sample_stride | 0 | optional uniform frame sampling stride for "
    "multimodal video tasks }"
    "{ max_frames | 0 | optional cap on sampled frames for multimodal video "
    "tasks }"
    "{ tokenizer_vocab | | path to tokenizer vocab.json for open-vocabulary "
    "detection }"
    "{ tokenizer_merges | | path to tokenizer merges.txt for open-vocabulary "
    "detection }"
    "{ bert_tokenizer_vocab | | path to BERT vocab.txt for Grounding DINO }"
    "{ weights w  | <none>  | path to models weights}"
    "{ use-gpu   | false  | activate gpu support}"
    "{ min_confidence | 0.25   | optional min confidence}"
    "{ nms_threshold  | 0.45   | NMS IoU threshold (YOLO-based "
    "detectors/segmenters) }"
    "{ mask_threshold | 0.50   | Mask binarization threshold (instance "
    "segmentation) }"
    "{ segmentation_output so | mask | segmentation representation: mask or "
    "polygon }"
    "{ batch b | 1 | Batch size}"
    "{ input_sizes is | | Input sizes for each model input. Format: "
    "CHW;CHW;... (e.g., '3,224,224' for single input or '3,224,224;3,224,224' "
    "for two inputs, '3,640,640;2' for rtdetr/dfine models) }"
    "{ warmup     | false  | enable GPU warmup}"
    "{ benchmark  | false  | enable benchmarking}"
    "{ export_metadata | false | print model metadata and exit without "
    "inference}"
    "{ no_gif  | false  | disable GIF output}"
    "{ iterations | 10     | number of iterations for benchmarking}"
    "{ num_frames nf | 0   | number of frames for video classification (0 = "
    "use model default, e.g., 16 for VideoMAE)}"
    "{ kserve_endpoint | | KServe V2 endpoint URL (e.g. http://127.0.0.1:8080) "
    "}"
    "{ kserve_model_name | | model name on the KServe endpoint (defaults to "
    "--type) }"
    "{ kserve_model_version | 1 | model version for KServe requests }"
    "{ kserve_timeout_ms | 30000 | KServe request timeout in milliseconds }"
    "{ kserve_transport | grpc | KServe transport: grpc (default) or http }"
    "{ input_mode im | preprocessed | input transport: preprocessed or "
    "encoded-image }"
    "{ task_model tm | | inner model used for task metadata in encoded-image "
    "mode }"
    "{ postprocess_mode pm | cpu | postprocessing placement: cpu or gpu }"
    "{ task_model_version tmv | 1 | version of --task_model }"
    "{ timings_csv | | write per-inference latency measurements to this CSV "
    "path }"
    "{ output_video | | write annotated output video to this path }"
    "{ no_display | false | do not open the preview window }";

namespace {

std::string lowered(const std::string &value) {
  std::string result = value;
  std::transform(
      result.begin(), result.end(), result.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

// Guard rails for the server-side ensemble path.
// Each one exists because the combination it rejects cannot work: the encoded
// image carries no tensor shape, the server decides batching, and GPU
// postprocessing only exists on the ensemble path.
void validateEnsembleArguments(const cv::CommandLineParser &parser) {
  const std::string inputMode = lowered(parser.get<std::string>("input_mode"));
  if (inputMode != "preprocessed" && inputMode != "encoded-image") {
    LOG(ERROR)
        << "--input_mode must be either 'preprocessed' or 'encoded-image'";
    std::exit(1);
  }

  const std::string postprocessMode =
      lowered(parser.get<std::string>("postprocess_mode"));
  if (postprocessMode != "cpu" && postprocessMode != "gpu") {
    LOG(ERROR) << "--postprocess_mode must be either 'cpu' or 'gpu'";
    std::exit(1);
  }

  if (inputMode == "encoded-image") {
    if (parser.get<std::string>("kserve_endpoint").empty()) {
      LOG(ERROR) << "--input_mode=encoded-image requires --kserve_endpoint";
      std::exit(1);
    }
    if (parser.get<std::string>("task_model").empty()) {
      LOG(ERROR) << "--task_model is required when --input_mode=encoded-image";
      std::exit(1);
    }
    if (parser.get<int>("batch") != 1) {
      LOG(ERROR) << "--input_mode=encoded-image currently requires --batch=1";
      std::exit(1);
    }
    if (parser.has("input_sizes")) {
      LOG(ERROR) << "--input_sizes must not be set with "
                    "--input_mode=encoded-image";
      std::exit(1);
    }
  }

  if (postprocessMode == "gpu" && inputMode != "encoded-image") {
    LOG(ERROR) << "--postprocess_mode=gpu requires --input_mode=encoded-image";
    std::exit(1);
  }
}

// A source the writer sink cannot serve. Deliberately wider than the router's
// still-image predicate (.jpg/.png): any common still-image extension is
// refused here, since none of those paths produce frames a video writer could
// take, and a still routed to the video loop by accident is not a video either.
bool isStillImageSource(const std::string &path) {
  const std::string extension = lowered(getFileExtension(path));
  return extension == "jpg" || extension == "jpeg" || extension == "png" ||
         extension == "bmp" || extension == "tiff";
}

} // namespace

AppConfig CommandLineParser::parseCommandLineArguments(int argc, char *argv[]) {
  cv::CommandLineParser parser(argc, argv, params);
  parser.about("Detect objects from video or image input source");

  if (parser.has("help")) {
    printHelpMessage(parser);
    std::exit(1);
  }

  if (parser.get<bool>("capabilities")) {
    if (!parser.check()) {
      parser.printErrors();
      std::exit(1);
    }
    AppConfig config;
    config.show_capabilities = true;
    return config;
  }

  validateArguments(parser);

  AppConfig config;
  std::string source_str = parser.get<std::string>("source");
  config.sources = split(source_str, ',');
  config.use_gpu = parser.get<bool>("use-gpu");
  config.enable_warmup = parser.get<bool>("warmup");
  config.enable_benchmark = parser.get<bool>("benchmark");
  config.export_metadata = parser.get<bool>("export_metadata");
  config.no_gif = parser.get<bool>("no_gif");
  config.benchmark_iterations = parser.get<int>("iterations");
  config.confidenceThreshold = parser.get<float>("min_confidence");
  config.nmsThreshold = parser.get<float>("nms_threshold");
  config.maskThreshold = parser.get<float>("mask_threshold");
  config.segmentationOutput = parser.get<std::string>("segmentation_output");
  std::transform(
      config.segmentationOutput.begin(), config.segmentationOutput.end(),
      config.segmentationOutput.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  config.detectorType = parser.get<std::string>("type");
  config.weights = parser.get<std::string>("weights");
  config.labelsPath = parser.get<std::string>("labels");
  config.tokenizerVocabPath = parser.get<std::string>("tokenizer_vocab");
  config.tokenizerMergesPath = parser.get<std::string>("tokenizer_merges");
  config.bertTokenizerVocabPath =
      parser.get<std::string>("bert_tokenizer_vocab");
  config.mmprojectPath = parser.get<std::string>("mmproj");
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
      std::transform(
          outputFormat.begin(), outputFormat.end(), outputFormat.begin(),
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
  if (parser.has("input_sizes")) {
    LOG(INFO) << "Parsing input sizes..."
              << parser.get<std::string>("input_sizes") << std::endl;
    input_sizes = parseInputSizes(parser.get<std::string>("input_sizes"));
    // Output the parsed sizes
    LOG(INFO) << "Parsed input sizes:\n";
    for (const auto &size : input_sizes) {
      LOG(INFO) << "(";
      for (size_t i = 0; i < size.size(); ++i) {
        LOG(INFO) << size[i];
        if (i < size.size() - 1) {
          LOG(INFO) << ",";
        }
      }
      LOG(INFO) << ")\n";
    }
  } else {
    LOG(INFO)
        << "No input sizes provided. Will use default model configuration."
        << std::endl;
  }
  // copy input sizes to config
  config.input_sizes = input_sizes;

  // Parse num_frames for video classification
  config.num_frames = parser.get<int>("num_frames");
  if (config.num_frames > 0) {
    LOG(INFO) << "Using " << config.num_frames
              << " frames for video classification";
  }

  config.kserve_endpoint = parser.get<std::string>("kserve_endpoint");
  config.kserve_model_name = parser.get<std::string>("kserve_model_name");
  config.kserve_model_version = parser.get<std::string>("kserve_model_version");
  config.kserve_timeout_ms = parser.get<int>("kserve_timeout_ms");
  config.kserve_transport = parser.get<std::string>("kserve_transport");
  std::transform(config.kserve_transport.begin(), config.kserve_transport.end(),
                 config.kserve_transport.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });

  config.input_mode = parser.get<std::string>("input_mode");
  config.task_model = parser.get<std::string>("task_model");
  config.postprocess_mode = parser.get<std::string>("postprocess_mode");
  config.task_model_version = parser.get<std::string>("task_model_version");
  std::transform(config.input_mode.begin(), config.input_mode.end(),
                 config.input_mode.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  std::transform(config.postprocess_mode.begin(), config.postprocess_mode.end(),
                 config.postprocess_mode.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });

  config.timings_csv = parser.get<std::string>("timings_csv");
  config.output_video = parser.get<std::string>("output_video");
  config.no_display = parser.get<bool>("no_display");

  if (!config.kserve_endpoint.empty() && config.kserve_model_name.empty()) {
    config.kserve_model_name = config.detectorType;
  }

  return config;
}

void CommandLineParser::printHelpMessage(const cv::CommandLineParser &parser) {
  parser.printMessage();
}

void CommandLineParser::validateArguments(const cv::CommandLineParser &parser) {
  if (!parser.check()) {
    parser.printErrors();
    std::exit(1);
  }

  const std::string typeForSourceCheck =
      normalizeModelType(parser.get<std::string>("type"));
  const bool is_text_task =
      (typeForSourceCheck == "gemma4" || typeForSourceCheck == "gemma" ||
       typeForSourceCheck == "llama" || typeForSourceCheck == "llamacpp" ||
       typeForSourceCheck == "imageunderstanding");
  const bool is_metadata_export = parser.get<bool>("export_metadata");

  std::string source = parser.get<std::string>("source");
  if (source.empty() && !is_text_task && !is_metadata_export) {
    LOG(ERROR) << "--source is required: pass an image or video path "
                  "(or use --export_metadata / a text task).";
    std::exit(1);
  }

  // Validate each source file exists (when provided)
  if (!source.empty()) {
    std::vector<std::string> sources = split(source, ',');
    for (const auto &src : sources) {
      if (!isFile(src) && !isDirectory(src)) {
        LOG(ERROR) << "Source file/directory " << src << " doesn't exist";
        std::exit(1);
      }
    }
  }

  std::string weights = parser.get<std::string>("weights");
  const std::string kserve_endpoint =
      parser.get<std::string>("kserve_endpoint");
  if (!kserve_endpoint.empty()) {
    if (weights.empty()) {
      weights = "kserve://" + kserve_endpoint;
    }
    if (parser.get<int>("kserve_timeout_ms") <= 0) {
      LOG(ERROR) << "--kserve_timeout_ms must be > 0";
      std::exit(1);
    }
    std::string transport = parser.get<std::string>("kserve_transport");
    std::transform(
        transport.begin(), transport.end(), transport.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (transport != "grpc" && transport != "http") {
      LOG(ERROR) << "--kserve_transport must be either 'grpc' or 'http'";
      std::exit(1);
    }
  } else if (!isFile(weights)) {
    LOG(ERROR) << "Weights file " << weights << " doesn't exist";
    std::exit(1);
  }

  std::string labelsPath = parser.get<std::string>("labels");
  if (!labelsPath.empty() && !isFile(labelsPath)) {
    LOG(ERROR) << "Labels file " << labelsPath << " doesn't exist";
    std::exit(1);
  }

  const std::string mmproj = parser.get<std::string>("mmproj");
  if (!mmproj.empty() && !isFile(mmproj)) {
    LOG(ERROR) << "Multimodal projector file " << mmproj << " doesn't exist";
    std::exit(1);
  }

  const std::string detectorType = parser.get<std::string>("type");
  const std::string normalizedType = normalizeModelType(detectorType);
  std::string outputFormat = parser.get<std::string>("output_format");
  const int sampleStride = parser.get<int>("sample_stride");
  const int maxFrames = parser.get<int>("max_frames");

  std::transform(
      outputFormat.begin(), outputFormat.end(), outputFormat.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (normalizedType == "owlv2" || normalizedType == "owlvit") {
    const std::string textPrompts = parser.get<std::string>("text_prompts");
    const std::string tokenizerVocab =
        parser.get<std::string>("tokenizer_vocab");
    const std::string tokenizerMerges =
        parser.get<std::string>("tokenizer_merges");

    if (textPrompts.empty()) {
      LOG(ERROR) << "Open-vocabulary detection requires --text_prompts";
      std::exit(1);
    }
    if (tokenizerVocab.empty()) {
      LOG(ERROR) << "Open-vocabulary detection requires --tokenizer_vocab";
      std::exit(1);
    }
    if (!isFile(tokenizerVocab)) {
      LOG(ERROR) << "Tokenizer vocab file " << tokenizerVocab
                 << " doesn't exist";
      std::exit(1);
    }
    if (tokenizerMerges.empty()) {
      LOG(ERROR) << "Open-vocabulary detection requires --tokenizer_merges";
      std::exit(1);
    }
    if (!isFile(tokenizerMerges)) {
      LOG(ERROR) << "Tokenizer merges file " << tokenizerMerges
                 << " doesn't exist";
      std::exit(1);
    }
  }

  if (normalizedType == "groundingdino") {
    const std::string textPrompts = parser.get<std::string>("text_prompts");
    const std::string bertVocab =
        parser.get<std::string>("bert_tokenizer_vocab");

    if (textPrompts.empty()) {
      LOG(ERROR) << "Grounding DINO requires --text_prompts";
      std::exit(1);
    }
    if (bertVocab.empty()) {
      LOG(ERROR) << "Grounding DINO requires --bert_tokenizer_vocab";
      std::exit(1);
    }
    if (!isFile(bertVocab)) {
      LOG(ERROR) << "BERT tokenizer vocab file " << bertVocab
                 << " doesn't exist";
      std::exit(1);
    }
  }

  if (!outputFormat.empty() && outputFormat != "text" &&
      outputFormat != "json") {
    LOG(ERROR) << "--output_format must be either 'text' or 'json'";
    std::exit(1);
  }

  std::string segmentationOutput =
      parser.get<std::string>("segmentation_output");
  std::transform(segmentationOutput.begin(), segmentationOutput.end(),
                 segmentationOutput.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (segmentationOutput != "mask" && segmentationOutput != "polygon") {
    LOG(ERROR) << "--segmentation_output must be either 'mask' or 'polygon'";
    std::exit(1);
  }

  validateEnsembleArguments(parser);

  if (sampleStride < 0) {
    LOG(ERROR) << "--sample_stride must be >= 0";
    std::exit(1);
  }

  if (maxFrames < 0) {
    LOG(ERROR) << "--max_frames must be >= 0";
    std::exit(1);
  }

  const std::string outputVideo = parser.get<std::string>("output_video");
  if (!outputVideo.empty()) {
    // An image source never enters the frame loops, so the writer would sit
    // unused while the run reports success -- reject it up front instead.
    for (const auto &src : split(source, ',')) {
      if (isStillImageSource(src)) {
        LOG(ERROR) << "--output_video applies to video sources only; image "
                      "sources write still results";
        std::exit(1);
      }
    }
#ifndef VIDEOCAPTURE_WITH_WRITER
    // Not a silent-ignore path: without the writer module there is nowhere for
    // the frames to go, and a run that asked for a file must not get none.
    LOG(ERROR) << "--output_video requires a build with "
                  "-DNEURIPLO_INFER_WITH_VIDEOWRITER=ON";
    std::exit(1);
#endif
  }
}
