#include "NeuriploInferApp.hpp"

#include "CLICommands.hpp"
#include "InferencePipeline.hpp"

#include <filesystem>
#include <stdexcept>

NeuriploInferApp::NeuriploInferApp(const AppConfig &app_config) : config(app_config) {
  setupLogging();
}

int NeuriploInferApp::run() {
  try {
    auto pipeline = InferencePipelineBuilder(config)
                        .source(config.sources)
                        .batch(config.batch_size)
                        .renderer(std::make_unique<DefaultResultRenderer>())
                        .build();

    if (config.export_metadata) {
      ExportMetadataCommand command;
      return command.execute(pipeline);
    }

    RunInferenceCommand command;
    return command.execute(pipeline);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    throw;
  }
}

void NeuriploInferApp::setupLogging(const std::string &log_folder) {
  try {
    if (!std::filesystem::exists(log_folder)) {
      std::filesystem::create_directory(log_folder);
    } else {
      std::filesystem::directory_iterator end_itr;
      for (std::filesystem::directory_iterator itr(log_folder); itr != end_itr;
           ++itr) {
        std::filesystem::remove(itr->path());
      }
    }

    google::InitGoogleLogging("object_detection");
    google::SetLogDestination(google::GLOG_INFO,
                              (log_folder + "/log_info_").c_str());
    google::SetLogDestination(google::GLOG_WARNING,
                              (log_folder + "/log_warning_").c_str());
    google::SetLogDestination(google::GLOG_ERROR,
                              (log_folder + "/log_error_").c_str());
    google::SetStderrLogging(google::GLOG_INFO);

    FLAGS_logbufsecs = 0;
    FLAGS_max_log_size = 100;
    FLAGS_stop_logging_if_full_disk = true;
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    throw;
  }
}
