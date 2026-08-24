#include "Capabilities.hpp"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#ifndef NEURIPLO_INFER_VERSION
#define NEURIPLO_INFER_VERSION "unknown"
#endif

namespace {
using json = nlohmann::json;

struct ModelCapability {
  std::string id;
  std::vector<std::string> aliases;
  std::vector<std::string> patterns;
  std::vector<std::string> required_parameters;
  std::vector<std::string> optional_parameters;
};

json parameterSelection(std::vector<std::string> required,
                        std::vector<std::string> optional) {
  return {{"required", std::move(required)}, {"optional", std::move(optional)}};
}

json modelCapability(const ModelCapability &model) {
  return {{"id", model.id},
          {"aliases", model.aliases},
          {"patterns", model.patterns},
          {"parameters", parameterSelection(model.required_parameters,
                                            model.optional_parameters)}};
}

json taskCapability(const std::string &id,
                    const std::vector<ModelCapability> &models,
                    std::vector<std::string> source_types, int min_sources,
                    int max_sources, std::vector<std::string> optional_params) {
  json serialized_models = json::array();
  for (const auto &model : models) {
    serialized_models.push_back(modelCapability(model));
  }

  return {{"id", id},
          {"models", std::move(serialized_models)},
          {"sources",
           {{"types", std::move(source_types)},
            {"min_items", min_sources},
            {"max_items", max_sources}}},
          {"parameters", parameterSelection({}, std::move(optional_params))}};
}

#ifdef NEURIPLO_INFER_WITH_LOCAL_BACKENDS
std::string normalizeBackendName(std::string backend) {
  std::transform(backend.begin(), backend.end(), backend.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  return backend;
}
#endif

std::vector<std::string> kserveTransports() {
  std::vector<std::string> transports{"http"};
#ifdef KSERVE_CLIENT_WITH_GRPC
  transports.emplace_back("grpc");
#endif
  return transports;
}

std::string defaultKserveTransport() {
#ifdef KSERVE_CLIENT_WITH_GRPC
  return "grpc";
#else
  return "http";
#endif
}

json parameterCatalog() {
  return {
      {"weights", {{"cli_flag", "weights"}, {"value_type", "path"}}},
      {"labels", {{"cli_flag", "labels"}, {"value_type", "path"}}},
      {"text_prompts",
       {{"cli_flag", "text_prompts"},
        {"value_type", "string_list"},
        {"separator", ";"}}},
      {"prompt", {{"cli_flag", "prompt"}, {"value_type", "string"}}},
      {"output_format",
       {{"cli_flag", "output_format"},
        {"value_type", "enum"},
        {"values", {"text", "json"}}}},
      {"sample_stride",
       {{"cli_flag", "sample_stride"},
        {"value_type", "integer"},
        {"minimum", 0}}},
      {"max_frames",
       {{"cli_flag", "max_frames"}, {"value_type", "integer"}, {"minimum", 0}}},
      {"tokenizer_vocab",
       {{"cli_flag", "tokenizer_vocab"}, {"value_type", "path"}}},
      {"tokenizer_merges",
       {{"cli_flag", "tokenizer_merges"}, {"value_type", "path"}}},
      {"bert_tokenizer_vocab",
       {{"cli_flag", "bert_tokenizer_vocab"}, {"value_type", "path"}}},
      {"mmproj", {{"cli_flag", "mmproj"}, {"value_type", "path"}}},
      {"min_confidence",
       {{"cli_flag", "min_confidence"},
        {"value_type", "number"},
        {"default", 0.25},
        {"minimum", 0.0},
        {"maximum", 1.0}}},
      {"nms_threshold",
       {{"cli_flag", "nms_threshold"},
        {"value_type", "number"},
        {"default", 0.45},
        {"minimum", 0.0},
        {"maximum", 1.0}}},
      {"mask_threshold",
       {{"cli_flag", "mask_threshold"},
        {"value_type", "number"},
        {"default", 0.5},
        {"minimum", 0.0},
        {"maximum", 1.0}}},
      {"segmentation_output",
       {{"cli_flag", "segmentation_output"},
        {"value_type", "enum"},
        {"default", "mask"},
        {"values", {"mask", "polygon"}}}},
      {"batch",
       {{"cli_flag", "batch"},
        {"value_type", "integer"},
        {"default", 1},
        {"minimum", 1}}},
      {"input_sizes",
       {{"cli_flag", "input_sizes"}, {"value_type", "shape_list"}}},
      {"use_gpu",
       {{"cli_flag", "use-gpu"},
        {"value_type", "boolean"},
        {"default", false}}},
      {"warmup",
       {{"cli_flag", "warmup"}, {"value_type", "boolean"}, {"default", false}}},
      {"benchmark",
       {{"cli_flag", "benchmark"},
        {"value_type", "boolean"},
        {"default", false}}},
      {"iterations",
       {{"cli_flag", "iterations"},
        {"value_type", "integer"},
        {"default", 10},
        {"minimum", 1}}},
      {"num_frames",
       {{"cli_flag", "num_frames"},
        {"value_type", "integer"},
        {"default", 0},
        {"minimum", 0}}},
      {"kserve_endpoint",
       {{"cli_flag", "kserve_endpoint"}, {"value_type", "url"}}},
      {"kserve_model_name",
       {{"cli_flag", "kserve_model_name"}, {"value_type", "string"}}},
      {"kserve_model_version",
       {{"cli_flag", "kserve_model_version"},
        {"value_type", "string"},
        {"default", "1"}}},
      {"kserve_transport",
       {{"cli_flag", "kserve_transport"},
        {"value_type", "enum"},
        {"default", defaultKserveTransport()},
        {"values", kserveTransports()}}},
      {"kserve_timeout_ms",
       {{"cli_flag", "kserve_timeout_ms"},
        {"value_type", "integer"},
        {"default", 30000},
        {"minimum", 1}}},
      {"input_mode",
       {{"cli_flag", "input_mode"},
        {"value_type", "enum"},
        {"default", "preprocessed"},
        {"values", {"preprocessed", "encoded-image"}}}},
      {"task_model", {{"cli_flag", "task_model"}, {"value_type", "string"}}},
      {"task_model_version",
       {{"cli_flag", "task_model_version"},
        {"value_type", "string"},
        {"default", "1"}}},
      {"postprocess_mode",
       {{"cli_flag", "postprocess_mode"},
        {"value_type", "enum"},
        {"default", "cpu"},
        {"values", {"cpu", "gpu"}}}},
  };
}

json executionWorkflows() {
  json workflows = json::array();

#ifdef NEURIPLO_INFER_WITH_LOCAL_BACKENDS
  workflows.push_back(
      {{"id", "local"},
       {"backends", {normalizeBackendName(NEURIPLO_INFER_LOCAL_BACKEND)}},
       {"protocols", json::array()},
       {"parameters",
        parameterSelection({"weights"},
                           {"use_gpu", "warmup", "benchmark", "iterations"})}});
#endif

#ifdef NEURIPLO_INFER_WITH_KSERVE
  workflows.push_back(
      {{"id", "client_server"},
       {"backends", json::array()},
       {"protocols",
        {{{"id", "kserve_v2"}, {"transports", kserveTransports()}}}},
       {"parameters",
        parameterSelection({"kserve_endpoint"},
                           {"kserve_model_name", "kserve_model_version",
                            "kserve_transport", "kserve_timeout_ms",
                            "input_mode", "task_model", "task_model_version",
                            "postprocess_mode", "benchmark", "iterations"})}});
#endif

  return workflows;
}

json taskCapabilities() {
  const std::vector<std::string> image_or_video{"image", "video"};

  return json::array(
      {taskCapability(
           "object_detection",
           {{"yolo",
             {"yolov4", "yolov7e2e", "yolov10", "yolo26", "yolonas"},
             {"yolo*"},
             {},
             {}},
            {"rtdetr", {"rtdetrul", "rtdetrultralytics"}, {}, {}, {}},
            {"rfdetr", {}, {}, {}, {}},
            {"ecdet", {"edgecrafter", "edgecrafter-det"}, {"ecdet*"}, {}, {}}},
           image_or_video, 1, 1,
           {"labels", "min_confidence", "nms_threshold", "batch",
            "input_sizes"}),
       taskCapability("instance_segmentation",
                      {{"yoloseg",
                        {"yolo-seg", "yolov8-seg", "yolov10seg", "yolo26seg"},
                        {"yolo*seg*"},
                        {},
                        {}},
                       {"rfdetrseg", {}, {}, {}, {}},
                       {"ecseg", {}, {"ecseg*", "edgecrafter*seg*"}, {}, {}}},
                      image_or_video, 1, 1,
                      {"labels", "min_confidence", "nms_threshold",
                       "mask_threshold", "segmentation_output", "batch",
                       "input_sizes"}),
       taskCapability("classification",
                      {{"torchvision-classifier", {}, {"resnet*"}, {}, {}},
                       {"tensorflow-classifier", {}, {"*tensorflow*"}, {}, {}},
                       {"vit-classifier", {}, {}, {}, {}}},
                      image_or_video, 1, 1, {"labels", "batch", "input_sizes"}),
       taskCapability("video_classification",
                      {{"videomae", {}, {}, {}, {}},
                       {"vivit", {}, {}, {}, {}},
                       {"timesformer", {}, {}, {}, {}}},
                      {"video"}, 1, 1,
                      {"labels", "num_frames", "batch", "input_sizes"}),
       taskCapability("optical_flow", {{"raft", {}, {}, {}, {}}}, {"image"}, 2,
                      -1, {"input_sizes"}),
       taskCapability(
           "pose_estimation",
           {{"yolov8pose", {"yolov8-pose"}, {}, {}, {}},
            {"yolo11pose", {"yolo11-pose"}, {}, {}, {}},
            {"yolo26pose", {"yolo26-pose"}, {}, {}, {}},
            {"yolov5pose", {"yolov5-pose"}, {}, {}, {}},
            {"vitpose", {}, {}, {}, {}},
            {"rfdetrpose",
             {"rfdetr-pose", "rfdetrkeypoint", "rfdetr-keypoint", "rfdetrkpt",
              "rfdetr-kpt"},
             {"rfdetr*keypoint*", "rfdetr*kpt*"},
             {},
             {}},
            {"ecpose", {}, {"ecpose*", "edgecrafter*pose*"}, {}, {}}},
           image_or_video, 1, 1,
           {"labels", "min_confidence", "nms_threshold", "batch",
            "input_sizes"}),
       taskCapability(
           "depth_estimation",
           {{"depth_anything_v2", {"depth-anything-v2"}, {}, {}, {}},
            {"yolo-depth", {"yolo26n-depth"}, {"yolo*depth*"}, {}, {}}},
           image_or_video, 1, 1, {"batch", "input_sizes"}),
       taskCapability(
           "open_vocabulary_detection",
           {{"owlv2",
             {"owlvit"},
             {},
             {"text_prompts", "tokenizer_vocab", "tokenizer_merges"},
             {}},
            {"groundingdino",
             {},
             {},
             {"text_prompts", "bert_tokenizer_vocab"},
             {}}},
           image_or_video, 1, 1,
           {"min_confidence", "nms_threshold", "batch", "input_sizes"}),
       taskCapability("image_understanding",
                      {{"gemma4",
                        {"gemma", "llama", "llamacpp", "imageunderstanding"},
                        {},
                        {},
                        {"mmproj"}}},
                      image_or_video, 0, 1,
                      {"prompt", "output_format", "sample_stride", "max_frames",
                       "input_sizes"}),
       taskCapability("gaussian_splatting",
                      {{"lgm", {"lgm-mini"}, {}, {}, {}},
                       {"grm", {}, {}, {}, {}},
                       {"gaussiansplatting", {}, {"*splat*"}, {}, {}}},
                      {"image"}, 1, 1, {"input_sizes"})});
}
} // namespace

nlohmann::json buildCapabilities() {
  return {{"schema_version", 1},
          {"producer",
           {{"name", "neuriplo-infer"}, {"version", NEURIPLO_INFER_VERSION}}},
          {"execution", {{"workflows", executionWorkflows()}}},
          {"source_types",
           {{{"id", "image"}, {"input", "file_path"}},
            {{"id", "video"}, {"input", "file_path"}}}},
          {"parameters", parameterCatalog()},
          {"tasks", taskCapabilities()}};
}

void printCapabilities(std::ostream &output) {
  output << buildCapabilities().dump(2) << '\n';
}
