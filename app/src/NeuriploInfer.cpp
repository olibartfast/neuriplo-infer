#include "NeuriploInfer.hpp"

#include "CLICommands.hpp"
#include "InferencePipeline.hpp"

#include <filesystem>
#include <stdexcept>

NeuriploInfer::NeuriploInfer(const AppConfig &app_config) : config(app_config) {
  setupLogging();
}

int NeuriploInfer::run() {
  try {
    auto pipeline = InferencePipelineBuilder(config)
                        .source(config.sources)
                        .batch(config.batch_size)
                        .renderer(std::make_unique<DefaultResultRenderer>())
                        .report(report)
                        .build();

    const int code = config.export_metadata
                         ? ExportMetadataCommand().execute(pipeline)
                         : RunInferenceCommand().execute(pipeline);
    if (code != 0) {
      report.fail(report.currentStage(),
                  "neuriplo-infer exited with code " + std::to_string(code));
    }
    neuriplo_infer::writeRunReport(report,
                                   neuriplo_infer::RunReport::kDefaultPath);
    return code;
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    // The stage the run had reached is the stage that failed; nothing here
    // classifies the message, which stays the producer's own.
    report.fail(report.currentStage(), e.what());
    neuriplo_infer::writeRunReport(report,
                                   neuriplo_infer::RunReport::kDefaultPath);
    throw;
  } catch (...) {
    // Backends do throw non-std types (OpenCV DNN throws `const char *` on
    // some ONNX graphs), which terminates the process. Termination stays the
    // behavior; the stage it died in is recorded on the way out.
    report.fail(report.currentStage(), {});
    neuriplo_infer::writeRunReport(report,
                                   neuriplo_infer::RunReport::kDefaultPath);
    throw;
  }
}

void NeuriploInfer::setupLogging(const std::string &log_folder) {
  try {
    if (!std::filesystem::exists(log_folder)) {
      std::filesystem::create_directories(log_folder);
    } else {
      std::filesystem::directory_iterator end_itr;
      for (std::filesystem::directory_iterator itr(log_folder); itr != end_itr;
           ++itr) {
        std::filesystem::remove(itr->path());
      }
    }

    if (!google::IsGoogleLoggingInitialized()) {
      google::InitGoogleLogging("object_detection");
    }
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
