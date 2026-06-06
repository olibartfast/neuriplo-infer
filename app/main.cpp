#include "CommandLineParser.hpp"
#include "NeuriploInferApp.hpp"
int main(int argc, char *argv[]) {
  try {
    AppConfig config = CommandLineParser::parseCommandLineArguments(argc, argv);
    NeuriploInferApp app(config);
    return app.run();
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    return 1;
  }
  return 0;
}