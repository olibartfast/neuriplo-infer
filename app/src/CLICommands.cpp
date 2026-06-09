#include "CLICommands.hpp"

#include "VideoCaptureFactory.hpp"
#include "neuriplo/tasks/core/opencv_interop.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>

namespace {

template <typename T1, typename T2>
std::vector<neuriplo_tasks::Tensor> convertToTensors(const T1 &outputs,
                                                     const T2 &shapes) {
  std::vector<neuriplo_tasks::Tensor> tensors;
  tensors.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    tensors.emplace_back(outputs[i], shapes[i]);
  }
  return tensors;
}

std::tuple<int, int, int, int>
extractInputDims(const std::vector<int64_t> &shape) {
  if (shape.size() == 4) {
    return {static_cast<int>(shape[0]), static_cast<int>(shape[1]),
            static_cast<int>(shape[2]), static_cast<int>(shape[3])};
  }
  if (shape.size() == 3) {
    return {1, static_cast<int>(shape[0]), static_cast<int>(shape[1]),
            static_cast<int>(shape[2])};
  }
  throw std::runtime_error(
      "Invalid input shape: expected 3D (CHW) or 4D (NCHW) tensor");
}

bool hasImageSources(const std::vector<std::string> &sources) {
  for (const auto &src : sources) {
    if (src.find(".jpg") != std::string::npos ||
        src.find(".png") != std::string::npos) {
      return true;
    }
  }
  return false;
}

void processImage(InferencePipeline &pipeline, const std::string &source) {
  cv::Mat image = cv::imread(source);
  if (pipeline.config.enable_warmup) {
    LOG(INFO) << "Warmup...";
    WarmupCommand warmup(image);
    warmup.execute(pipeline);
  }

  auto start = std::chrono::steady_clock::now();
  const auto &first_input = pipeline.inference_metadata.getInputs()[0];
  auto [batch, channels, height, width] = extractInputDims(first_input.shape);

  LOG(INFO) << "Model input shape: " << batch << "x" << channels << "x"
            << height << "x" << width;
  LOG(INFO) << "Image dimensions: " << image.rows << "x" << image.cols << "x"
            << image.channels();

  const auto preprocessed = pipeline.task->preprocess({image});
  const auto [outputs, shapes] =
      pipeline.engine->get_infer_results(preprocessed);

  auto tensors = convertToTensors(outputs, shapes);
  auto results = pipeline.task->postprocess(image.size(), tensors);
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "Inference time: " << duration << " ms";

  pipeline.renderResults(results, image);
  std::filesystem::create_directories("data/output");
  std::string processed_path = "data/output/processed.png";
  if (!cv::imwrite(processed_path, image)) {
    const std::string fallback_path = "/tmp/neuriplo-infer-processed.png";
    if (!cv::imwrite(fallback_path, image)) {
      LOG(ERROR) << "Failed to save output image to both " << processed_path
                 << " and " << fallback_path;
    } else {
      LOG(WARNING) << "Could not write " << processed_path
                   << ", saved output to " << fallback_path;
    }
  } else {
    LOG(INFO) << "Saved processed image to: " << processed_path;
  }

  if (pipeline.config.enable_benchmark) {
    BenchmarkCommand benchmark(image);
    benchmark.execute(pipeline);
  }
}

void processVideo(InferencePipeline &pipeline, const std::string &source) {
  std::unique_ptr<VideoCaptureInterface> videoInterface =
      createVideoInterface();
  if (!videoInterface->initialize(source)) {
    throw std::runtime_error("Failed to initialize video capture for input: " +
                             source);
  }

  cv::Mat frame;
  while (videoInterface->readFrame(frame)) {
    auto start = std::chrono::steady_clock::now();
    const auto preprocessed = pipeline.task->preprocess({frame});
    const auto [outputs, shapes] =
        pipeline.engine->get_infer_results(preprocessed);

    auto tensors = convertToTensors(outputs, shapes);
    auto results = pipeline.task->postprocess(frame.size(), tensors);
    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    double fps = 1000.0 / static_cast<double>(duration);
    std::string fpsText = "FPS: " + std::to_string(fps);
    cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);

    pipeline.renderResults(results, frame);

    cv::imshow("opencv feed", frame);
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q') {
      LOG(INFO) << "Exit requested";
      break;
    }
  }

  videoInterface->release();
}

void processVideoClassification(InferencePipeline &pipeline,
                                const std::string &source) {
  std::unique_ptr<VideoCaptureInterface> videoInterface =
      createVideoInterface();
  if (!videoInterface->initialize(source)) {
    throw std::runtime_error("Failed to initialize video capture for input: " +
                             source);
  }

  const int requiredFrames = pipeline.getRequiredFrameCount();
  LOG(INFO) << "Video classification mode: accumulating " << requiredFrames
            << " frames";

  cv::Mat frame;
  std::vector<cv::Mat> frameBuffer;
  frameBuffer.reserve(static_cast<size_t>(requiredFrames));

  while (videoInterface->readFrame(frame)) {
    frameBuffer.push_back(frame.clone());

    if (static_cast<int>(frameBuffer.size()) >= requiredFrames) {
      auto start = std::chrono::steady_clock::now();
      const auto preprocessed = pipeline.task->preprocess(frameBuffer);
      const auto [outputs, shapes] =
          pipeline.engine->get_infer_results(preprocessed);

      auto tensors = convertToTensors(outputs, shapes);
      auto results = pipeline.task->postprocess(frame.size(), tensors);
      auto end = std::chrono::steady_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      double fps = 1000.0 / static_cast<double>(duration);
      std::string fpsText = "FPS: " + std::to_string(fps);

      cv::Mat displayFrame = frameBuffer.back().clone();
      cv::putText(displayFrame, fpsText, cv::Point(10, 30),
                  cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

      pipeline.renderResults(results, displayFrame);
      cv::imshow("opencv feed", displayFrame);

      frameBuffer.erase(frameBuffer.begin());
    }

    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q') {
      LOG(INFO) << "Exit requested";
      break;
    }
  }

  videoInterface->release();
}

void processOpticalFlow(InferencePipeline &pipeline) {
  LOG(INFO) << "Processing optical flow for image pairs";

  for (size_t i = 0; i < pipeline.config.sources.size() - 1; i++) {
    std::vector<std::string> flowInputs = {pipeline.config.sources[i],
                                           pipeline.config.sources[i + 1]};
    std::vector<cv::Mat> images;
    for (const auto &name : flowInputs) {
      cv::Mat img = cv::imread(name);
      if (img.empty()) {
        LOG(ERROR) << "Could not open or read the image: " << name;
        continue;
      }
      images.push_back(img);
    }

    if (images.size() != 2) {
      continue;
    }

    auto start = std::chrono::steady_clock::now();
    const auto preprocessed = pipeline.task->preprocess(images);
    auto [infer_results, infer_shapes] =
        pipeline.engine->get_infer_results(preprocessed);

    auto tensors = convertToTensors(infer_results, infer_shapes);
    auto predictions = pipeline.task->postprocess(
        cv::Size(images[0].cols, images[0].rows), tensors);

    auto end = std::chrono::steady_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    LOG(INFO) << "Infer time for " << images.size() << " images: " << diff
              << " ms";

    cv::Mat &image = images[0];
    for (const auto &prediction : predictions) {
      if (std::holds_alternative<neuriplo_tasks::OpticalFlow>(prediction)) {
        neuriplo_tasks::OpticalFlow flow =
            std::get<neuriplo_tasks::OpticalFlow>(prediction);
        neuriplo_tasks::toCvMat(flow.flow).copyTo(image);
      }
    }

    std::string sourceDir =
        flowInputs[0].substr(0, flowInputs[0].find_last_of("/\\"));
    std::string outputDir = sourceDir + "/output";
    std::filesystem::create_directories(outputDir);
    std::string processedFrameFilename =
        outputDir + "/processed_frame_optical_flow.jpg";
    LOG(INFO) << "Saving frame to: " << processedFrameFilename;
    cv::imwrite(processedFrameFilename, image);
  }
}

void processImageUnderstanding(InferencePipeline &pipeline) {
  const std::string prompt_log =
      pipeline.config.taskExtraParams.count("prompt")
          ? pipeline.config.taskExtraParams.at("prompt")
          : "(default)";
  LOG(INFO) << "Running image understanding with prompt: " << prompt_log;

  std::vector<cv::Mat> images;
  if (!pipeline.config.sources.empty() && !pipeline.config.sources[0].empty()) {
    cv::Mat img = cv::imread(pipeline.config.sources[0]);
    if (img.empty()) {
      LOG(WARNING) << "Could not read source image: "
                   << pipeline.config.sources[0] << " - running text-only";
    } else {
      LOG(INFO) << "Source image: " << pipeline.config.sources[0] << " ("
                << img.cols << "x" << img.rows << ")";
      images.push_back(std::move(img));
    }
  }

  const auto preprocessed = pipeline.task->preprocess(images);

  auto start = std::chrono::steady_clock::now();
  const auto [outputs, shapes] =
      pipeline.engine->get_infer_results(preprocessed);
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "Inference time: " << duration << " ms";

  auto tensors = convertToTensors(outputs, shapes);
  cv::Mat dummy;
  auto results = pipeline.task->postprocess(cv::Size{0, 0}, tensors);
  pipeline.renderResults(results, dummy);
}

std::string taskTypeName(neuriplo_tasks::TaskType task_type) {
  switch (task_type) {
  case neuriplo_tasks::TaskType::OpticalFlow:
    return "OpticalFlow";
  case neuriplo_tasks::TaskType::Classification:
    return "Classification";
  case neuriplo_tasks::TaskType::Detection:
    return "Detection";
  case neuriplo_tasks::TaskType::InstanceSegmentation:
    return "InstanceSegmentation";
  case neuriplo_tasks::TaskType::VideoClassification:
    return "VideoClassification";
  case neuriplo_tasks::TaskType::PoseEstimation:
    return "PoseEstimation";
  case neuriplo_tasks::TaskType::DepthEstimation:
    return "DepthEstimation";
  case neuriplo_tasks::TaskType::OpenVocabDetection:
    return "OpenVocabDetection";
  case neuriplo_tasks::TaskType::GaussianSplatting:
    return "GaussianSplatting";
  case neuriplo_tasks::TaskType::ImageUnderstanding:
    return "ImageUnderstanding";
  }
  return "Unknown";
}

void printLayerList(const char *label, const std::vector<LayerInfo> &layers) {
  std::cout << label << ":\n";
  for (const auto &layer : layers) {
    std::cout << "  " << layer.name << " shape=[";
    for (size_t i = 0; i < layer.shape.size(); ++i) {
      std::cout << layer.shape[i];
      if (i + 1 < layer.shape.size()) {
        std::cout << ",";
      }
    }
    std::cout << "] batch_size=" << layer.batch_size << '\n';
  }
}

} // namespace

WarmupCommand::WarmupCommand(cv::Mat image) : image_(std::move(image)) {}

int WarmupCommand::execute(InferencePipeline &pipeline) {
  for (int i = 0; i < 5; ++i) {
    const auto preprocessed = pipeline.task->preprocess({image_});
    const auto [outputs, shapes] =
        pipeline.engine->get_infer_results(preprocessed);

    auto tensors = convertToTensors(outputs, shapes);
    auto results = pipeline.task->postprocess(image_.size(), tensors);
    (void)results;
  }
  return 0;
}

BenchmarkCommand::BenchmarkCommand(cv::Mat image) : image_(std::move(image)) {}

int BenchmarkCommand::execute(InferencePipeline &pipeline) {
  double total_time = 0.0;
  for (int i = 0; i < pipeline.config.benchmark_iterations; ++i) {
    auto start = std::chrono::steady_clock::now();

    const auto preprocessed = pipeline.task->preprocess({image_});
    const auto [outputs, shapes] =
        pipeline.engine->get_infer_results(preprocessed);

    auto tensors = convertToTensors(outputs, shapes);
    auto results = pipeline.task->postprocess(image_.size(), tensors);
    (void)results;

    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    LOG(INFO) << "Iteration " << i << ": " << duration << "ms";
    total_time += static_cast<double>(duration);
  }
  double average_time =
      total_time / static_cast<double>(pipeline.config.benchmark_iterations);
  LOG(INFO) << "Average inference time over "
            << pipeline.config.benchmark_iterations
            << " iterations: " << average_time << "ms";
  return 0;
}

int RunInferenceCommand::execute(InferencePipeline &pipeline) {
  if (pipeline.task_type == neuriplo_tasks::TaskType::ImageUnderstanding) {
    processImageUnderstanding(pipeline);
    return 0;
  }

  if (hasImageSources(pipeline.config.sources)) {
    if (pipeline.config.sources.size() == 1) {
      processImage(pipeline, pipeline.config.sources[0]);
      return 0;
    }
    if (pipeline.config.sources.size() >= 2 &&
        pipeline.task_type == neuriplo_tasks::TaskType::OpticalFlow) {
      processOpticalFlow(pipeline);
      return 0;
    }
    LOG(ERROR) << "Multiple image sources only supported for optical flow";
    throw std::runtime_error(
        "Multiple image sources only supported for optical flow");
  }

  if (pipeline.config.sources.size() != 1) {
    LOG(ERROR) << "Video processing requires single source";
    throw std::runtime_error("Video processing requires single source");
  }

  if (pipeline.task_type == neuriplo_tasks::TaskType::VideoClassification) {
    processVideoClassification(pipeline, pipeline.config.sources[0]);
  } else {
    processVideo(pipeline, pipeline.config.sources[0]);
  }
  return 0;
}

int ExportMetadataCommand::execute(InferencePipeline &pipeline) {
  std::cout << "model_type: " << pipeline.config.detectorType << '\n';
  std::cout << "task_type: " << taskTypeName(pipeline.task_type) << '\n';
  printLayerList("inputs", pipeline.inference_metadata.getInputs());
  printLayerList("outputs", pipeline.inference_metadata.getOutputs());
  return 0;
}
