#pragma once

#include "AppConfig.hpp"

#include <string>

class VisionApp {
public:
  explicit VisionApp(const AppConfig &config);
  int run();

private:
  void setupLogging(const std::string &log_folder = "./logs");

  AppConfig config;
};
