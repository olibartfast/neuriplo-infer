#pragma once

#include "VideoCaptureFactory.hpp"
#include "DetectorSetup.hpp"
#include "InferenceBackendSetup.hpp"
#include "utils.hpp"
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

class ObjectDetectionApp {
public:
    ObjectDetectionApp(const AppConfig& config);
    void run();

    void setDetector(std::unique_ptr<Detector> newDetector) {
        detector = std::move(newDetector);
    }

    void setEngine(std::unique_ptr<InferenceInterface> newEngine) {
        engine = std::move(newEngine);
    }

private:
    void warmup_gpu(const cv::Mat& image);
    void benchmark(const cv::Mat& image);
    void setupLogging(const std::string& log_folder = "./logs");
    void processImage(const std::string& source);
    void processVideo(const std::string& source);

    AppConfig config;
    std::unique_ptr<InferenceInterface> engine;
    std::unique_ptr<Detector> detector;
    std::vector<std::string> classes;
};

AppConfig parseCommandLineArguments(int argc, char *argv[]);
