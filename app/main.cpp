#include "Capabilities.hpp"
#include "CommandLineParser.hpp"
#include "NeuriploInfer.hpp"
#include "RunReport.hpp"

#include <glog/logging.h>
#include <iostream>

int main(int argc, char *argv[]) {
  // Initialize logging before argument parsing so validation errors are
  // emitted through glog instead of the uninitialized-logger stderr path.
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(argv[0]);
  }
  AppConfig config;
  // Argument validation exits the process directly, so the report for that
  // stage is armed here rather than written from a catch block.
  neuriplo_infer::armConfigurationExitReport(
      neuriplo_infer::RunReport::kDefaultPath);
  try {
    config = CommandLineParser::parseCommandLineArguments(argc, argv);
    neuriplo_infer::disarmConfigurationExitReport();
    if (config.show_capabilities) {
      printCapabilities(std::cout);
      return 0;
    }
  } catch (const std::exception &e) {
    // A run that never started still owes the caller an attributable reason.
    neuriplo_infer::disarmConfigurationExitReport();
    LOG(ERROR) << "Error: " << e.what();
    neuriplo_infer::writeConfigurationFailureReport(
        e.what(), neuriplo_infer::RunReport::kDefaultPath);
    return 1;
  }

  try {
    NeuriploInfer app(config);
    return app.run();
  } catch (const std::exception &e) {
    // run() has already written its report with the failing stage.
    LOG(ERROR) << "Error: " << e.what();
    return 1;
  }
}
