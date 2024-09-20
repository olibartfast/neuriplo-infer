#pragma once
#include "common.hpp"

struct AppConfig {
    std::string detectorType;
    std::string source;
    std::string labelsPath;
    std::string config;
    std::string weights;
    bool use_gpu;
    bool enable_warmup;
    bool enable_benchmark;
    int benchmark_iterations;
    float confidenceThreshold;
};

