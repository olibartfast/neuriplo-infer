#include "CLICommands.hpp"

#include "FrameConversion.hpp"
#include "VideoCaptureFactory.hpp"
#ifdef VIDEOCAPTURE_WITH_WRITER
#include "VideoWriterConfig.hpp"
#include "VideoWriterFactory.hpp"
#endif
#include "neuriplo/tasks/core/opencv_interop.hpp"
#ifdef NEURIPLO_INFER_WITH_KSERVE
#include "KserveEngine.hpp"
#endif

#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {

neuriplo_tasks::vision::Image toTaskImage(const cv::Mat &image) {
  return neuriplo_tasks::vision::opencv::copyFromCvMat(image);
}

std::vector<neuriplo_tasks::vision::Image>
toTaskImages(const std::vector<cv::Mat> &images) {
  std::vector<neuriplo_tasks::vision::Image> converted;
  converted.reserve(images.size());
  for (const auto &image : images) {
    converted.push_back(toTaskImage(image));
  }
  return converted;
}

neuriplo_tasks::vision::Size toTaskSize(const cv::Mat &image) {
  return {image.cols, image.rows};
}

// Replaces characters that are awkward in filenames with '-' so model/backend
// tags can be embedded in the output image name safely.
std::string sanitizeForFilename(std::string value) {
  for (auto &c : value) {
    if (std::isalnum(static_cast<unsigned char>(c)) == 0) {
      c = '-';
    }
  }
  return value;
}

// Short tag for the backend a KServe server used to run the model, derived from
// the V2 metadata "platform" string (e.g. "tensorrt_plan" -> "trt"). Returns ""
// when the platform is unknown so the caller falls back to a plain "kserve".
std::string kservePlatformTag(const std::string &platform) {
  std::string lower = platform;
  for (auto &c : lower) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (lower.empty()) {
    return "";
  }
  if (lower.find("tensorrt") != std::string::npos) {
    return "trt";
  }
  if (lower.find("onnx") != std::string::npos) {
    return "ort";
  }
  if (lower.find("openvino") != std::string::npos) {
    return "openvino";
  }
  // Check litert/tflite before the generic tensorflow match: the neuriplo
  // KServe runtime reports "neuriplo_litert" for TFLite models.
  if (lower.find("litert") != std::string::npos ||
      lower.find("tflite") != std::string::npos) {
    return "litert";
  }
  if (lower.find("torch") != std::string::npos) {
    return "torch";
  }
  if (lower.find("tensorflow") != std::string::npos) {
    return "tf";
  }
  if (lower.find("python") != std::string::npos) {
    return "python";
  }
  // Unrecognised but non-empty platform: keep it (sanitised) rather than drop
  // the information.
  return sanitizeForFilename(lower);
}

// Short label for the execution backend, used in the output image filename.
// Remote inference reports "kserve" plus the served backend when the server
// advertises one (e.g. "kserve_trt"); local inference reports the compile-time
// backend selected via DEFAULT_BACKEND.
std::string backendLabel(const InferencePipeline &pipeline) {
  if (!pipeline.config.kserve_endpoint.empty()) {
    const std::string tag = kservePlatformTag(pipeline.kserve_platform);
    return tag.empty() ? "kserve" : "kserve_" + tag;
  }
#if defined(USE_ONNX_RUNTIME)
  return "onnx_runtime";
#elif defined(USE_LIBTORCH)
  return "libtorch";
#elif defined(USE_TENSORRT)
  return "tensorrt";
#elif defined(USE_OPENVINO)
  return "openvino";
#elif defined(USE_EXECUTORCH)
  return "executorch";
#elif defined(USE_LIBTENSORFLOW)
  return "libtensorflow";
#elif defined(USE_OPENCV_DNN)
  return "opencv_dnn";
#elif defined(USE_LITERT)
  return "litert";
#else
  return "local";
#endif
}

// data/output/processed_<model>_<backend>.png
std::string processedImageName(const InferencePipeline &pipeline) {
  return "processed_" + sanitizeForFilename(pipeline.config.detectorType) +
         "_" + sanitizeForFilename(backendLabel(pipeline)) + ".png";
}

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

// Runs one frame through the configured path and returns task results.
//
// Three shapes, chosen by configuration:
//   preprocessed            local preprocess -> infer -> local postprocess
//   encoded-image, cpu      encoded bytes -> server preprocess+infer -> local
//   postprocess encoded-image, gpu      encoded bytes -> server does everything
//   -> decode envelope
//
// The two encoded modes send the compressed file instead of a dense float
// tensor, which is the whole point: a 640x640x3 FP32 tensor is ~4.9 MB, a JPEG
// of the same frame is a few dozen KB.
std::vector<neuriplo_tasks::Result>
inferFrame(InferencePipeline &pipeline, const cv::Mat &frame,
           const std::vector<uint8_t> *encoded_source) {
  using neuriplo_infer::RunStage;
  using neuriplo_infer::StageTimer;

  if (!pipeline.encoded_image) {
    // Each stage is timed where it happens; the lambdas keep the deduced
    // contract types out of this file.
    const auto preprocessed = [&] {
      StageTimer timer(pipeline.report, RunStage::Preprocess);
      return pipeline.task->preprocess({toTaskImage(frame)});
    }();
    const auto inferred = [&] {
      StageTimer timer(pipeline.report, RunStage::Inference);
      return pipeline.engine->get_infer_results(preprocessed);
    }();
    const auto &[outputs, shapes] = inferred;
    StageTimer timer(pipeline.report, RunStage::Postprocess);
    auto tensors = convertToTensors(outputs, shapes);
    return pipeline.task->postprocess(toTaskSize(frame), tensors);
  }

#ifdef NEURIPLO_INFER_WITH_KSERVE
  // Video frames arrive decoded, so they are re-encoded here; still image
  // sources pass their original file bytes through untouched.
  std::vector<uint8_t> encoded;
  if (encoded_source != nullptr) {
    encoded = *encoded_source;
  } else if (!cv::imencode(".jpg", frame, encoded)) {
    throw std::runtime_error("Could not JPEG-encode a frame for the ensemble");
  }

  const std::vector<std::vector<uint8_t>> request = {std::move(encoded)};
  const auto inferred = [&] {
    StageTimer timer(pipeline.report, RunStage::Inference);
    return pipeline.engine->get_infer_results(request);
  }();
  const auto &[outputs, shapes] = inferred;

  StageTimer postprocess_timer(pipeline.report, RunStage::Postprocess);
  if (!pipeline.server_postprocess) {
    auto tensors = convertToTensors(outputs, shapes);
    return pipeline.task->postprocess(toTaskSize(frame), tensors);
  }

  auto *kserve = dynamic_cast<KserveEngine *>(pipeline.engine.get());
  if (kserve == nullptr) {
    throw std::runtime_error(
        "server-side postprocessing requires the KServe engine");
  }
  return neuriplo_infer::decodeEnvelope(kserve->lastRawOutputs(),
                                        pipeline.envelope_variant, frame.cols,
                                        frame.rows);
#else
  (void)encoded_source;
  throw std::runtime_error(
      "--input_mode=encoded-image needs a build with KServe support");
#endif
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
  cv::Mat image;
  {
    neuriplo_infer::StageTimer timer(pipeline.report,
                                     neuriplo_infer::RunStage::Source);
    image = cv::imread(source);
    // Reading nothing used to surface as a confusing failure further down the
    // pipeline; saying so here also attributes it to the source stage.
    if (image.empty()) {
      throw std::runtime_error("Could not read the image source: " + source);
    }
  }
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

  // In encoded-image mode the original file bytes go on the wire, so the source
  // is never decoded and re-encoded.
  std::vector<uint8_t> source_bytes;
  if (pipeline.encoded_image) {
    std::ifstream stream(source, std::ios::binary);
    if (!stream.is_open()) {
      throw std::runtime_error("Could not open encoded image: " + source);
    }
    source_bytes.assign(std::istreambuf_iterator<char>(stream), {});
  }

  auto results = inferFrame(pipeline, image,
                            pipeline.encoded_image ? &source_bytes : nullptr);
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "Inference time: " << duration << " ms";

  {
    // Drawing the results and writing the image are one stage: both exist only
    // to produce the artifact, and the benchmark below must stay out of it.
    neuriplo_infer::StageTimer render_timer(pipeline.report,
                                            neuriplo_infer::RunStage::Render);
    pipeline.renderResults(results, image);
    std::filesystem::create_directories("data/output");
    const std::string processed_name = processedImageName(pipeline);
    std::string processed_path = "data/output/" + processed_name;
    if (!cv::imwrite(processed_path, image)) {
      const std::string fallback_path = "/tmp/neuriplo-infer-" + processed_name;
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
  }

  if (pipeline.report != nullptr) {
    pipeline.report->addSample();
  }

  if (pipeline.config.enable_benchmark) {
    BenchmarkCommand benchmark(image);
    benchmark.execute(pipeline);
  }
}

// Per-inference latency, one row per inference, opened only when asked for.
// The run report already carries per-stage totals and throughput; what it
// cannot show is the distribution -- a run whose mean is fine because a slow
// first frame is averaged away by four hundred quick ones looks identical to a
// uniformly quick run. That distinction is the reason to record frames
// individually, so this writes alongside the report rather than duplicating it.
class FrameTimingsCsv {
public:
  // Opened before the first frame, not lazily: a run that cannot write its
  // measurements should fail before spending minutes producing them.
  explicit FrameTimingsCsv(const std::string &path) {
    if (path.empty()) {
      return;
    }
    const std::filesystem::path csv_path(path);
    if (csv_path.has_parent_path()) {
      std::filesystem::create_directories(csv_path.parent_path());
    }
    out_.open(csv_path);
    if (!out_.is_open()) {
      throw std::runtime_error("Could not open timings CSV for writing: " +
                               path);
    }
    path_ = path;
    out_ << "frame,latency_us\n";
  }

  void add(std::size_t frame_index, std::int64_t latency_us) {
    if (out_.is_open()) {
      out_ << frame_index << ',' << latency_us << '\n';
    }
  }

  void finish() {
    if (out_.is_open()) {
      out_.close();
      LOG(INFO) << "Saved per-inference timings to: " << path_;
    }
  }

private:
  std::ofstream out_;
  std::string path_;
};

#ifdef VIDEOCAPTURE_WITH_WRITER
// RAII over the videocapture writer: creates and initializes exactly once
// and guarantees release() on every exit path, so the destination is a
// complete playable file even when the run stops early.
class OutputVideoSink {
public:
  OutputVideoSink(const std::string &destination, int width, int height) {
    videocapture::VideoWriterConfig config;
    config.width = width;
    config.height = height;
    config.frameRate = 30.0;
    config.codec = videocapture::VideoCodec::Auto;
    writer_ = createVideoWriter();
    if (!writer_) {
      throw std::runtime_error(
          "--output_video: no writer backend available for destination: " +
          destination);
    }
    if (!writer_->initialize(destination, config)) {
      throw std::runtime_error(
          "--output_video: could not initialize the video writer for " +
          destination);
    }
  }
  ~OutputVideoSink() { writer_->release(); }
  OutputVideoSink(const OutputVideoSink &) = delete;
  OutputVideoSink &operator=(const OutputVideoSink &) = delete;
  // writeFrame false is a failed run, not a skippable frame.
  void write(videocapture::Frame frame, std::size_t frame_index) {
    if (!writer_->writeFrame(frame)) {
      throw std::runtime_error(
          "--output_video: video writer failed to write frame " +
          std::to_string(frame_index));
    }
  }

private:
  std::unique_ptr<VideoWriterInterface> writer_;
};
#endif

void processVideo(InferencePipeline &pipeline, const std::string &source) {
  std::unique_ptr<VideoCaptureInterface> videoInterface =
      createVideoInterface();
  {
    neuriplo_infer::StageTimer timer(pipeline.report,
                                     neuriplo_infer::RunStage::Source);
    if (!videoInterface->initialize(source)) {
      throw std::runtime_error(
          "Failed to initialize video capture for input: " + source);
    }
  }

  FrameTimingsCsv timings(pipeline.config.timings_csv);

#ifdef VIDEOCAPTURE_WITH_WRITER
  std::unique_ptr<OutputVideoSink> output_sink;
#endif

  videocapture::Frame frame;
  std::size_t frame_index = 0;
  bool read_to_end = false;
  while (true) {
    bool read_frame = false;
    {
      // Every frame read is source work. Reading in the loop condition would
      // attribute a decode failure half way through a video to whatever stage
      // the previous frame ended in.
      neuriplo_infer::StageTimer timer(pipeline.report,
                                       neuriplo_infer::RunStage::Source);
      read_frame = videoInterface->readFrame(frame);
    }
    if (!read_frame) {
      read_to_end = true;
      break;
    }

    // Bridged per read, not once outside the loop: the Mat aliases the
    // frame's storage, which readFrame may reallocate.
    cv::Mat image = neuriplo_infer::toBgrMat(frame);

#ifdef VIDEOCAPTURE_WITH_WRITER
    // Configured once from the first frame: video sources have fixed dims.
    if (!pipeline.config.output_video.empty() && !output_sink) {
      output_sink = std::make_unique<OutputVideoSink>(
          pipeline.config.output_video, image.cols, image.rows);
    }
#endif

    auto start = std::chrono::steady_clock::now();
    auto results = inferFrame(pipeline, image, nullptr);
    if (pipeline.report != nullptr) {
      pipeline.report->addFrames(1);
    }
    auto end = std::chrono::steady_clock::now();
    // Microseconds, not milliseconds: a frame faster than 1 ms truncated to
    // zero, and the overlay then divided by it -- inf FPS on exactly the
    // backends worth measuring.
    const auto latency_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    timings.add(frame_index, latency_us);
    const double fps =
        latency_us > 0 ? 1e6 / static_cast<double>(latency_us) : 0.0;
    std::string fpsText = "FPS: " + std::to_string(fps);

    {
      // Overlay, drawing, and display are the rendered output for a video, the
      // way the written PNG is for a still image.
      neuriplo_infer::StageTimer render_timer(pipeline.report,
                                              neuriplo_infer::RunStage::Render);
      cv::putText(image, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                  1, cv::Scalar(0, 255, 0), 2);
      pipeline.renderResults(results, image);
#ifdef VIDEOCAPTURE_WITH_WRITER
      if (output_sink) {
        output_sink->write(neuriplo_infer::toFrame(image), frame_index);
      }
#endif
      if (!pipeline.config.no_display) {
        cv::imshow("opencv feed", image);
      }
    }

    ++frame_index;

    // Only a shown window can be asked to close. Without one there is no
    // keypress to read, and cv::waitKey would spin the loop for nothing.
    if (!pipeline.config.no_display) {
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') {
        LOG(INFO) << "Exit requested";
        break;
      }
    }
  }

  timings.finish();
  videoInterface->release();
  // One video read to its end is one sample, however many frames it held. A
  // video the operator stopped early was not processed to completion, and
  // counting it would report a source the run never finished.
  if (read_to_end && pipeline.report != nullptr) {
    pipeline.report->addSample();
  }
}

void processVideoClassification(InferencePipeline &pipeline,
                                const std::string &source) {
  std::unique_ptr<VideoCaptureInterface> videoInterface =
      createVideoInterface();
  {
    neuriplo_infer::StageTimer timer(pipeline.report,
                                     neuriplo_infer::RunStage::Source);
    if (!videoInterface->initialize(source)) {
      throw std::runtime_error(
          "Failed to initialize video capture for input: " + source);
    }
  }

  const int requiredFrames = pipeline.getRequiredFrameCount();
  LOG(INFO) << "Video classification mode: accumulating " << requiredFrames
            << " frames";

  FrameTimingsCsv timings(pipeline.config.timings_csv);

#ifdef VIDEOCAPTURE_WITH_WRITER
  std::unique_ptr<OutputVideoSink> output_sink;
#endif

  videocapture::Frame frame;
  std::vector<cv::Mat> frameBuffer;
  frameBuffer.reserve(static_cast<size_t>(requiredFrames));

  // Counts frames read, so a row's index refers to the frame the window closed
  // on. One inference here consumes a whole window, so rows are sparser than in
  // processVideo -- which is why the column is the frame index and not a row
  // counter that would silently mean something different per mode.
  std::size_t frame_index = 0;
  bool read_to_end = false;
  while (true) {
    bool read_frame = false;
    {
      neuriplo_infer::StageTimer timer(pipeline.report,
                                       neuriplo_infer::RunStage::Source);
      read_frame = videoInterface->readFrame(frame);
    }
    if (!read_frame) {
      read_to_end = true;
      break;
    }
    // clone(): the window outlives the next read, and on the packed-BGR8
    // path the bridged Mat is only a view onto storage that read reuses.
    cv::Mat image = neuriplo_infer::toBgrMat(frame);
#ifdef VIDEOCAPTURE_WITH_WRITER
    // Configured once from the first frame: video sources have fixed dims.
    if (!pipeline.config.output_video.empty() && !output_sink) {
      output_sink = std::make_unique<OutputVideoSink>(
          pipeline.config.output_video, image.cols, image.rows);
    }
#endif
    frameBuffer.push_back(image.clone());
    ++frame_index;

    if (static_cast<int>(frameBuffer.size()) >= requiredFrames) {
      auto start = std::chrono::steady_clock::now();
      const auto preprocessed = [&] {
        neuriplo_infer::StageTimer timer(pipeline.report,
                                         neuriplo_infer::RunStage::Preprocess);
        return pipeline.task->preprocess(toTaskImages(frameBuffer));
      }();
      const auto inferred = [&] {
        neuriplo_infer::StageTimer timer(pipeline.report,
                                         neuriplo_infer::RunStage::Inference);
        return pipeline.engine->get_infer_results(preprocessed);
      }();
      const auto &[outputs, shapes] = inferred;

      auto results = [&] {
        neuriplo_infer::StageTimer timer(pipeline.report,
                                         neuriplo_infer::RunStage::Postprocess);
        auto tensors = convertToTensors(outputs, shapes);
        return pipeline.task->postprocess(toTaskSize(image), tensors);
      }();
      // One inference consumes the whole accumulated window.
      if (pipeline.report != nullptr) {
        pipeline.report->addFrames(requiredFrames);
      }
      auto end = std::chrono::steady_clock::now();
      const auto latency_us =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      timings.add(frame_index - 1, latency_us);
      const double fps =
          latency_us > 0 ? 1e6 / static_cast<double>(latency_us) : 0.0;
      std::string fpsText = "FPS: " + std::to_string(fps);

      {
        neuriplo_infer::StageTimer render_timer(
            pipeline.report, neuriplo_infer::RunStage::Render);
        cv::Mat displayFrame = frameBuffer.back().clone();
        cv::putText(displayFrame, fpsText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        pipeline.renderResults(results, displayFrame);
#ifdef VIDEOCAPTURE_WITH_WRITER
        if (output_sink) {
          // The accumulator's rendered window is the last read frame, so the
          // writer row follows the timings index.
          output_sink->write(neuriplo_infer::toFrame(displayFrame),
                             frame_index - 1);
        }
#endif
        if (!pipeline.config.no_display) {
          cv::imshow("opencv feed", displayFrame);
        }
      }

      frameBuffer.erase(frameBuffer.begin());
    }

    if (!pipeline.config.no_display) {
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') {
        LOG(INFO) << "Exit requested";
        break;
      }
    }
  }

  timings.finish();
  videoInterface->release();
  // Same rule as processVideo: only a video read to its end is a sample.
  if (read_to_end && pipeline.report != nullptr) {
    pipeline.report->addSample();
  }
}

void processOpticalFlow(InferencePipeline &pipeline) {
  LOG(INFO) << "Processing optical flow for image pairs";

  for (size_t i = 0; i < pipeline.config.sources.size() - 1; i++) {
    std::vector<std::string> flowInputs = {pipeline.config.sources[i],
                                           pipeline.config.sources[i + 1]};
    std::vector<cv::Mat> images;
    {
      neuriplo_infer::StageTimer timer(pipeline.report,
                                       neuriplo_infer::RunStage::Source);
      for (const auto &name : flowInputs) {
        cv::Mat img = cv::imread(name);
        if (img.empty()) {
          // A source the run was asked for and could not read is a failed run,
          // not a quiet skip: continuing here returned success with no samples
          // and no artifact, which is indistinguishable from having nothing to
          // do. Throwing inside the source scope attributes it to `source`.
          throw std::runtime_error("Could not open or read the image: " + name);
        }
        images.push_back(img);
      }
    }

    auto start = std::chrono::steady_clock::now();
    // Optical flow does not go through inferFrame, so it carries its own
    // timers; without them a flow run reports no stages at all and attributes
    // any failure to whatever stage ran last, which is the model load.
    const auto preprocessed = [&] {
      neuriplo_infer::StageTimer timer(pipeline.report,
                                       neuriplo_infer::RunStage::Preprocess);
      return pipeline.task->preprocess(toTaskImages(images));
    }();
    const auto inferred = [&] {
      neuriplo_infer::StageTimer timer(pipeline.report,
                                       neuriplo_infer::RunStage::Inference);
      return pipeline.engine->get_infer_results(preprocessed);
    }();
    const auto &[infer_results, infer_shapes] = inferred;

    auto predictions = [&] {
      neuriplo_infer::StageTimer timer(pipeline.report,
                                       neuriplo_infer::RunStage::Postprocess);
      auto tensors = convertToTensors(infer_results, infer_shapes);
      return pipeline.task->postprocess(toTaskSize(images[0]), tensors);
    }();

    auto end = std::chrono::steady_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    LOG(INFO) << "Infer time for " << images.size() << " images: " << diff
              << " ms";

    {
      neuriplo_infer::StageTimer render_timer(pipeline.report,
                                              neuriplo_infer::RunStage::Render);
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

    // A pair carried through to its written flow image is one sample.
    if (pipeline.report != nullptr) {
      pipeline.report->addSample();
    }
  }
}

void processImageUnderstanding(InferencePipeline &pipeline) {
  const std::string prompt_log =
      pipeline.config.taskExtraParams.count("prompt")
          ? pipeline.config.taskExtraParams.at("prompt")
          : "(default)";
  LOG(INFO) << "Running image understanding with prompt: " << prompt_log;

  std::vector<cv::Mat> images;
  {
    neuriplo_infer::StageTimer timer(pipeline.report,
                                     neuriplo_infer::RunStage::Source);
    if (!pipeline.config.sources.empty() &&
        !pipeline.config.sources[0].empty()) {
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
  }

  // Image understanding has its own path too, and needs the same timers for
  // the same reason optical flow does.
  const auto preprocessed = [&] {
    neuriplo_infer::StageTimer timer(pipeline.report,
                                     neuriplo_infer::RunStage::Preprocess);
    return pipeline.task->preprocess(toTaskImages(images));
  }();

  auto start = std::chrono::steady_clock::now();
  const auto inferred = [&] {
    neuriplo_infer::StageTimer timer(pipeline.report,
                                     neuriplo_infer::RunStage::Inference);
    return pipeline.engine->get_infer_results(preprocessed);
  }();
  const auto &[outputs, shapes] = inferred;
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "Inference time: " << duration << " ms";

  auto results = [&] {
    neuriplo_infer::StageTimer timer(pipeline.report,
                                     neuriplo_infer::RunStage::Postprocess);
    auto tensors = convertToTensors(outputs, shapes);
    return pipeline.task->postprocess(neuriplo_tasks::vision::Size{0, 0},
                                      tensors);
  }();

  {
    neuriplo_infer::StageTimer render_timer(pipeline.report,
                                            neuriplo_infer::RunStage::Render);
    cv::Mat dummy;
    pipeline.renderResults(results, dummy);
  }

  if (pipeline.report != nullptr) {
    pipeline.report->addSample();
  }
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
  // Warmup repeats the real path but produces nothing, so its time must stay
  // out of the run's stage totals: it would be counted without a matching
  // sample and would drag the reported throughput down with it.
  neuriplo_infer::TimingSuspension untimed(pipeline.report);
  // Goes through inferFrame so warmup exercises the configured transport. In
  // encoded-image mode the server expects a UINT8 IMAGE, and preprocessing
  // locally here would send it a dense float tensor instead.
  for (int i = 0; i < 5; ++i) {
    auto results = inferFrame(pipeline, image_, nullptr);
    (void)results;
  }
  return 0;
}

BenchmarkCommand::BenchmarkCommand(cv::Mat image) : image_(std::move(image)) {}

int BenchmarkCommand::execute(InferencePipeline &pipeline) {
  // Same reason as warmup: the benchmark measures the engine, and its own
  // numbers go to the log rather than into the run's stage totals.
  neuriplo_infer::TimingSuspension untimed(pipeline.report);
  double total_time = 0.0;
  for (int i = 0; i < pipeline.config.benchmark_iterations; ++i) {
    auto start = std::chrono::steady_clock::now();

    auto results = inferFrame(pipeline, image_, nullptr);
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
