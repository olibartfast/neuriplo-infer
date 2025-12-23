#pragma once
#include "CommandLineParser.hpp"
#include "InferenceBackendSetup.hpp"
#include "VideoCaptureFactory.hpp"
#include "utils.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/core/task_interface.hpp"

class ObjectDetectionApp {
public:
  ObjectDetectionApp(const AppConfig &config);
  void run();
  void setTask(std::unique_ptr<vision_core::TaskInterface> newTask) {
    task = std::move(newTask);
  }
  void setEngine(std::unique_ptr<InferenceInterface> newEngine) {
    engine = std::move(newEngine);
  }

private:
  void warmup_gpu(const cv::Mat &image);
  void benchmark(const cv::Mat &image);
  void setupLogging(const std::string &log_folder = "./logs");
  void processImage(const std::string &source);
  void processVideo(const std::string &source);
  std::tuple<int, int, int, int>
  extractInputDims(const std::vector<int64_t> &shape);

  AppConfig config;
  std::unique_ptr<InferenceInterface> engine;
  std::unique_ptr<vision_core::TaskInterface> task;
  std::vector<std::string> classes;
};
