#pragma once

#include "AppConfig.hpp"
#include "InferenceBackendSetup.hpp"
#include "ResultRenderer.hpp"
#include "TaskRouting.hpp"
#include "vision-core/core/model_info.hpp"
#include "vision-core/core/task_interface.hpp"

#include <memory>
#include <string>
#include <vector>

struct InferencePipeline {
  AppConfig config;
  std::unique_ptr<InferenceInterface> engine;
  std::unique_ptr<vision_core::TaskInterface> task;
  std::unique_ptr<ResultRenderer> renderer;
  std::vector<std::string> classes;
  InferenceMetadata inference_metadata;
  vision_core::ModelInfo model_info;
  vision_core::TaskType task_type{vision_core::TaskType::Detection};

  int getRequiredFrameCount() const;
  void renderResults(const std::vector<vision_core::Result> &results,
                     cv::Mat &image);
};

class InferencePipelineBuilder {
public:
  explicit InferencePipelineBuilder(const AppConfig &config);

  InferencePipelineBuilder &source(const std::vector<std::string> &sources);
  InferencePipelineBuilder &batch(int batch_size);
  InferencePipelineBuilder &renderer(std::unique_ptr<ResultRenderer> renderer);

  InferencePipeline build();

private:
  AppConfig config_;
  std::unique_ptr<ResultRenderer> renderer_;

  // Staged helpers for auditable pipeline construction
  void logPipelineConfig() const;
  void loadLabels(InferencePipeline &pipeline) const;
  void setupBackend(InferencePipeline &pipeline) const;
  void setupTask(InferencePipeline &pipeline) const;
  void setupPresentation(InferencePipeline &pipeline);
};
