#pragma once

#include "AppConfig.hpp"
#ifdef NEURIPLO_INFER_WITH_LOCAL_BACKENDS
// Brings in neuriplo's local-engine factory and, transitively, the inference
// contract (InferenceInterface / TensorElement / InferenceMetadata).
#include "InferenceBackendSetup.hpp"
#else
// KServe-only build: no neuriplo fetch, so use the app-local contract headers.
#include "InferenceInterface.hpp"
#include "InferenceMetadata.hpp"
#endif
#include "ResultRenderer.hpp"
#include "TaskRouting.hpp"
#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <memory>
#include <string>
#include <vector>

struct InferencePipeline {
  AppConfig config;
  std::unique_ptr<InferenceInterface> engine;
  std::unique_ptr<neuriplo_tasks::TaskInterface> task;
  std::unique_ptr<ResultRenderer> renderer;
  std::vector<std::string> classes;
  InferenceMetadata inference_metadata;
  neuriplo_tasks::ModelInfo model_info;
  neuriplo_tasks::TaskType task_type{neuriplo_tasks::TaskType::Detection};

  int getRequiredFrameCount() const;
  void renderResults(const std::vector<neuriplo_tasks::Result> &results,
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
