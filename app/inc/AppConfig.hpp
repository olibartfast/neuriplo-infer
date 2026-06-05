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
  int batch_size;
  std::vector<std::vector<int64_t>> input_sizes;
  int num_frames{
      0}; // Number of frames for video classification (0 = use model default)
  std::string kserve_endpoint;
  std::string kserve_model_name;
  std::string kserve_model_version{"1"};
  int kserve_timeout_ms{30000};
  std::string kserve_transport{"grpc"};
};
