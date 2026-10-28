#include "CLICommands.hpp"
#include "Capabilities.hpp"
#include "CommandLineParser.hpp"
#include "InferencePipeline.hpp"
#include "ModelInputTypes.hpp"
#include "PredictionRecorder.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <type_traits>

namespace {
namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace neuriplo_tasks;

using Recorder = neuriplo_infer::validation::PredictionRecorder;

class ObservedTask final : public TaskInterface {
public:
  ObservedTask(std::unique_ptr<TaskInterface> task, Recorder &recorder)
      : TaskInterface(task->getModelInfo()), task_(std::move(task)),
        recorder_(recorder) {}
  TaskType getTaskType() override { return task_->getTaskType(); }
  int getRequiredFrames() const override { return task_->getRequiredFrames(); }
  std::vector<std::vector<uint8_t>>
  preprocess(const std::vector<Image> &images) override {
    auto buffers = task_->preprocess(images);
    json inputs = json::array();
    for (const auto &buffer : buffers) {
      inputs.push_back(recorder_.bytes(buffer));
    }
    recorder_.document["preprocess_calls"].push_back(
        {{"images", images.size()}, {"inputs", inputs}});
    return buffers;
  }
  std::vector<Result> postprocess(const Size &size,
                                  const std::vector<Tensor> &outputs) override {
    auto results = task_->postprocess(size, outputs);
    recorder_.predictions(size, results, outputs);
    return results;
  }

private:
  std::unique_ptr<TaskInterface> task_;
  Recorder &recorder_;
};

class ObservedRenderer final : public DefaultResultRenderer {
public:
  explicit ObservedRenderer(Recorder &recorder) : recorder_(recorder) {}
  void render(const std::vector<Result> &results, cv::Mat &image,
              const RenderContext &context) override {
    recorder_.predictions(Size(image.cols, image.rows), results);
    DefaultResultRenderer::render(results, image, context);
  }

private:
  Recorder &recorder_;
};
} // namespace

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  try {
    const auto config =
        CommandLineParser::parseCommandLineArguments(argc, argv);
    if (config.show_capabilities) {
      printCapabilities(std::cout);
      return 0;
    }
    const char *directory = std::getenv("NEURIPLO_VALIDATION_OUTPUT");
    if (directory == nullptr || *directory == '\0') {
      throw std::runtime_error(
          "Set NEURIPLO_VALIDATION_OUTPUT to a new output directory");
    }
    Recorder recorder(directory);
    neuriplo_infer::RunReport report;
    auto pipeline = InferencePipelineBuilder(config)
                        .source(config.sources)
                        .batch(config.batch_size)
                        .report(report)
                        .build();
    json inputs = json::array();
    for (const auto &input : pipeline.inference_metadata.getInputs()) {
      inputs.push_back({{"name", input.name},
                        {"shape", input.shape},
                        {"dtype", neuriplo_infer::datatypeTag(input.datatype)},
                        {"batch", input.batch_size}});
    }
    recorder.document["model_inputs"] = inputs;
    recorder.document["task_inputs"] = {
        {"shapes", pipeline.model_info.input_shapes},
        {"datatypes", pipeline.model_info.input_datatypes},
        {"formats", pipeline.model_info.input_formats}};
    if (pipeline.task) {
      pipeline.task =
          std::make_unique<ObservedTask>(std::move(pipeline.task), recorder);
    }
    if (pipeline.server_postprocess) {
      pipeline.renderer = std::make_unique<ObservedRenderer>(recorder);
    }
    try {
      const int code = RunInferenceCommand().execute(pipeline);
      recorder.document["exit_code"] = code;
      recorder.document["run_report"] = report.toJson();
      recorder.flush();
      return code;
    } catch (...) {
      recorder.document["exit_code"] = 1;
      recorder.flush();
      throw;
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
