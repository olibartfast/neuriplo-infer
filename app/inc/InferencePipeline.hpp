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
#include "RunReport.hpp"
#include "TaskRouting.hpp"
#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <memory>
#include <string>
#include <vector>

#ifdef NEURIPLO_INFER_WITH_KSERVE
#include "KserveEnvelope.hpp"
#endif

struct InferencePipeline {
  AppConfig config;
  std::unique_ptr<InferenceInterface> engine;
  std::unique_ptr<neuriplo_tasks::TaskInterface> task;
  std::unique_ptr<ResultRenderer> renderer;
  std::vector<std::string> classes;
  InferenceMetadata inference_metadata;
  neuriplo_tasks::ModelInfo model_info;
  neuriplo_tasks::TaskType task_type{neuriplo_tasks::TaskType::Detection};
  // Serving backend reported by a remote KServe server (V2 metadata
  // "platform", e.g. "tensorrt_plan"). Empty for local engines or when the
  // server omits it. Used to tag the rendered output filename.
  std::string kserve_platform;

  // Diagnostics collector for this run, owned by the caller. Null when nobody
  // is collecting, which is what keeps tests and library users unaffected.
  neuriplo_infer::RunReport *report{nullptr};

  // Server-side ensemble mode. When encoded_image is set, frames go to the
  // server as encoded bytes instead of a preprocessed tensor. When
  // server_postprocess is also set, the server returns a decoded result
  // envelope and local postprocessing is skipped entirely.
  bool encoded_image{false};
  bool server_postprocess{false};
#ifdef NEURIPLO_INFER_WITH_KSERVE
  neuriplo_infer::EnvelopeVariant envelope_variant{
      neuriplo_infer::EnvelopeVariant::Detection};
#endif

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
  InferencePipelineBuilder &report(neuriplo_infer::RunReport &report);

  InferencePipeline build();

private:
  AppConfig config_;
  std::unique_ptr<ResultRenderer> renderer_;
  neuriplo_infer::RunReport *report_{nullptr};

  // Staged helpers for auditable pipeline construction
  void logPipelineConfig() const;
  void loadLabels(InferencePipeline &pipeline) const;
  void setupBackend(InferencePipeline &pipeline) const;
  void setupTask(InferencePipeline &pipeline) const;
  void setupPresentation(InferencePipeline &pipeline);
};
