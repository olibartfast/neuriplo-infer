#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "AppConfig.hpp"

class CommandLineParser {
public:
    static AppConfig parseCommandLineArguments(int argc, char *argv[]);

private:
    static const std::string params;
    static void printHelpMessage(const cv::CommandLineParser& parser);
    static void validateArguments(const cv::CommandLineParser& parser);
};
