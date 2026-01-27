#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <any>
#include <algorithm>
#include <iterator>
#include <type_traits> // for std::remove_pointer
#include <glog/logging.h>

struct AppConfig {
    std::string detectorType;
    std::vector<std::string> sources;
    std::string labelsPath;
    std::string weights;
    bool use_gpu;
    bool enable_warmup;
    bool enable_benchmark;
    int benchmark_iterations;
    float confidenceThreshold;
    int batch_size;
    std::vector<std::vector<int64_t>> input_sizes;
    int num_frames{0};  // Number of frames for video classification (0 = use model default)
};

