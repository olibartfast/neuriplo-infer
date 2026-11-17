#pragma once
#include "AppConfig.hpp"
#include "ArgParser.hpp"

class CommandLineParser {
public:
  static AppConfig parseCommandLineArguments(int argc, char *argv[]);

private:
  static const std::string params;
  static void printHelpMessage(const neuriplo_infer::ArgParser &parser);
  static void validateArguments(const neuriplo_infer::ArgParser &parser);
};
