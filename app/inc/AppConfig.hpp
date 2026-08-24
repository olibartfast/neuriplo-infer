#pragma once
#include <algorithm>
#include <any>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <type_traits> // for std::remove_pointer

struct AppConfig {
  bool show_capabilities{false};
  std::string detectorType;
  std::vector<std::string> sources;
  std::string labelsPath;
  std::string tokenizerVocabPath;
  std::string tokenizerMergesPath;
  std::string bertTokenizerVocabPath;
  std::string weights;
  std::string mmprojectPath;
  std::vector<std::string> textPrompts;
  std::map<std::string, std::string> taskExtraParams;
  bool use_gpu{false};
  bool enable_warmup{false};
  bool enable_benchmark{false};
  bool export_metadata{false};
  bool no_gif{false};
  int benchmark_iterations;
  float confidenceThreshold;
  float nmsThreshold{0.45f};
  float maskThreshold{0.50f};
  // Instance-segmentation representation: "mask" (default) or "polygon".
  std::string segmentationOutput{"mask"};
  int batch_size;
  std::vector<std::vector<int64_t>> input_sizes;
  int num_frames{
      0}; // Number of frames for video classification (0 = use model default)
  std::string kserve_endpoint;
  std::string kserve_model_name;
  std::string kserve_model_version{"1"};
  int kserve_timeout_ms{30000};
  std::string kserve_transport{"http"};
  // Server-side ensemble path. "preprocessed" sends a dense tensor the client
  // preprocessed; "encoded-image" sends the encoded file and lets the server
  // preprocess. task_model names the inner model whose metadata drives task
  // construction, since the ensemble's own metadata only describes an image.
  std::string input_mode{"preprocessed"};
  std::string task_model;
  // Version of --task_model. Separate from kserve_model_version because a graph
  // may reference an inner model version other than the ensemble's own.
  std::string task_model_version{"1"};
  std::string postprocess_mode{"cpu"};
};
