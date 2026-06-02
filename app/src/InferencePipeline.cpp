#include "InferencePipeline.hpp"

#include "utils.hpp"
#include "vision-core/core/task_config.hpp"
#include "vision-core/core/task_factory.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

std::string buildEngineWeights(const AppConfig &config) {
  std::string engine_weights = config.weights;
  if (!config.mmprojectPath.empty()) {
    engine_weights += "|mmproj=" + config.mmprojectPath;
  }
  return engine_weights;
}

void setInputFormat(vision_core::ModelInfo &model_info) {
  if (model_info.input_formats.empty() || model_info.input_shapes.empty() ||
      model_info.input_shapes[0].empty()) {
    return;
  }

  const auto &shape = model_info.input_shapes[0];
  if (shape.size() == 4) {
    const bool is_nchw = (shape[1] == 1 || shape[1] == 3);
    const bool is_nhwc = (shape[3] == 1 || shape[3] == 3);

    if (is_nchw && !is_nhwc) {
      model_info.input_formats[0] = "FORMAT_NCHW";
    } else if (!is_nchw && is_nhwc) {
      model_info.input_formats[0] = "FORMAT_NHWC";
    } else if (shape[2] > 3 && shape[3] > 3) {
      model_info.input_formats[0] = "FORMAT_NCHW";
    } else if (shape[1] > 3 && shape[2] > 3) {
      model_info.input_formats[0] = "FORMAT_NHWC";
    } else {
      model_info.input_formats[0] = "FORMAT_NCHW";
    }
  } else if (shape.size() == 3) {
    model_info.input_formats[0] = "FORMAT_NCHW";
  }
}

vision_core::ModelInfo
buildModelInfo(const InferenceMetadata &inference_metadata,
               const AppConfig &config) {
  vision_core::ModelInfo model_info;
  for (size_t i = 0; i < inference_metadata.getInputs().size(); i++) {
    const auto &input = inference_metadata.getInputs()[i];
    std::vector<int64_t> shape;

    if (i < config.input_sizes.size() && !config.input_sizes[i].empty()) {
      for (auto dim : config.input_sizes[i]) {
        shape.push_back(dim);
      }
      if (shape.size() == 3) {
        shape.insert(shape.begin(), config.batch_size);
      }
    } else {
      shape = input.shape;
      if (shape.size() == 3) {
        shape.insert(shape.begin(), config.batch_size);
      }
    }

    model_info.addInput(input.name, shape, input.batch_size);
  }

  for (const auto &output : inference_metadata.getOutputs()) {
    model_info.addOutput(output.name, output.shape, output.batch_size);
  }

  setInputFormat(model_info);
  if (!model_info.input_types.empty()) {
    model_info.input_types[0] = CV_32F;
  }
  return model_info;
}

std::string readFile(const std::string &path, const std::string &label) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("Can't open " + label + " file: " + path);
  }
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

vision_core::TaskConfig buildTaskConfig(const AppConfig &config) {
  vision_core::TaskConfig task_config;
  task_config.confidence_threshold = config.confidenceThreshold;
  task_config.nms_threshold = config.nmsThreshold;
  task_config.mask_threshold = config.maskThreshold;
  task_config.text_prompts = config.textPrompts;
  task_config.extra_params = config.taskExtraParams;

  if (!config.tokenizerVocabPath.empty()) {
    task_config.tokenizer_vocab_json =
        readFile(config.tokenizerVocabPath, "tokenizer vocab");
  }
  if (!config.tokenizerMergesPath.empty()) {
    task_config.tokenizer_merges_text =
        readFile(config.tokenizerMergesPath, "tokenizer merges");
  }
  if (!config.bertTokenizerVocabPath.empty()) {
    task_config.bert_tokenizer_vocab_text =
        readFile(config.bertTokenizerVocabPath, "BERT tokenizer vocab");
  }

  return task_config;
}

} // namespace

int InferencePipeline::getRequiredFrameCount() const {
  if (config.num_frames > 0) {
    return config.num_frames;
  }
  return task ? task->getRequiredFrames() : 1;
}

void InferencePipeline::renderResults(
    const std::vector<vision_core::Result> &results, cv::Mat &image) {
  RenderContext context{task_type, classes, config.confidenceThreshold};
  renderer->render(results, image, context);
}

InferencePipelineBuilder::InferencePipelineBuilder(const AppConfig &config)
    : config_(config) {}

InferencePipelineBuilder &
InferencePipelineBuilder::source(const std::vector<std::string> &sources) {
  config_.sources = sources;
  return *this;
}

InferencePipelineBuilder &InferencePipelineBuilder::backend() { return *this; }

InferencePipelineBuilder &InferencePipelineBuilder::task() { return *this; }

InferencePipelineBuilder &
InferencePipelineBuilder::precision(const std::string & /*precision*/) {
  return *this;
}

InferencePipelineBuilder &InferencePipelineBuilder::batch(int batch_size) {
  config_.batch_size = batch_size;
  return *this;
}

InferencePipelineBuilder &
InferencePipelineBuilder::renderer(std::unique_ptr<ResultRenderer> renderer) {
  renderer_ = std::move(renderer);
  return *this;
}

InferencePipeline InferencePipelineBuilder::build() {
  InferencePipeline pipeline;
  pipeline.config = config_;

  LOG(INFO) << "Sources: ";
  for (const auto &src : pipeline.config.sources) {
    LOG(INFO) << " " << src;
  }
  LOG(INFO) << "Weights " << pipeline.config.weights;
  LOG(INFO) << "Labels file " << pipeline.config.labelsPath;
  LOG(INFO) << "Detector type " << pipeline.config.detectorType;
  if (!pipeline.config.textPrompts.empty()) {
    LOG(INFO) << "Open-vocab prompts count "
              << pipeline.config.textPrompts.size();
  }
  if (!pipeline.config.taskExtraParams.empty()) {
    LOG(INFO) << "Task extra params count "
              << pipeline.config.taskExtraParams.size();
  }

  if (!pipeline.config.labelsPath.empty()) {
    pipeline.classes = readLabelNames(pipeline.config.labelsPath);
  }

  LOG(INFO) << "CPU info " << getCPUInfo();
  LOG(INFO) << "GPU info: " << getGPUModel();
  const auto use_gpu = pipeline.config.use_gpu && hasNvidiaGPU();
  pipeline.engine = setup_inference_engine(buildEngineWeights(pipeline.config),
                                           use_gpu, pipeline.config.batch_size,
                                           pipeline.config.input_sizes);
  if (!pipeline.engine) {
    throw std::runtime_error("Can't setup an inference engine for " +
                             pipeline.config.weights);
  }

  pipeline.inference_metadata = pipeline.engine->get_inference_metadata();
  pipeline.model_info =
      buildModelInfo(pipeline.inference_metadata, pipeline.config);
  pipeline.task_type = getTaskTypeForModel(pipeline.config.detectorType);

  LOG(INFO) << "Using vision-core model type: " << pipeline.config.detectorType;
  pipeline.task = vision_core::TaskFactory::createTaskInstance(
      pipeline.config.detectorType, pipeline.model_info,
      buildTaskConfig(pipeline.config));
  if (!pipeline.task) {
    throw std::runtime_error("Can't setup a task for " +
                             pipeline.config.detectorType);
  }

  pipeline.renderer = renderer_ ? std::move(renderer_)
                                : std::make_unique<DefaultResultRenderer>();
  return pipeline;
}
