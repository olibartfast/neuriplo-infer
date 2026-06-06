#pragma once

#include "AppConfig.hpp"

#include <string>

class NeuriploInferApp {
public:
  explicit NeuriploInferApp(const AppConfig &app_config);
  int run();

private:
  void setupLogging(const std::string &log_folder = "./logs");

  AppConfig config;
};
