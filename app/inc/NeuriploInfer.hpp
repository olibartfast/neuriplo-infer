#pragma once

#include "AppConfig.hpp"

#include <string>

class NeuriploInfer {
public:
  explicit NeuriploInfer(const AppConfig &app_config);
  int run();

private:
  void setupLogging(const std::string &log_folder = "./logs");

  AppConfig config;
};
