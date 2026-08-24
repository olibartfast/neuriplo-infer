#include "Capabilities.hpp"
#include "CommandLineParser.hpp"
#include "NeuriploInfer.hpp"

#include <glog/logging.h>
#include <iostream>

int main(int argc, char *argv[]) {
  // Initialize logging before argument parsing so validation errors are
  // emitted through glog instead of the uninitialized-logger stderr path.
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(argv[0]);
  }
  try {
    AppConfig config = CommandLineParser::parseCommandLineArguments(argc, argv);
    if (config.show_capabilities) {
      printCapabilities(std::cout);
      return 0;
    }
    NeuriploInfer app(config);
    return app.run();
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    return 1;
  }
  return 0;
}
