#pragma once
#include "VideoCaptureFactory.hpp"
#include "DetectorSetup.hpp"
#include "InferenceBackendSetup.hpp"
#include "utils.hpp"
#include "common.hpp"
#include "CommandLineParser.hpp"
#include "Detector.hpp"



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

