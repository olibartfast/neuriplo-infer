#include "VisionApp.hpp"

#include <chrono>
#include <filesystem>

namespace {

// Helper to convert raw outputs and shapes to vision_core::Tensor objects
template<typename T1, typename T2>
std::vector<vision_core::Tensor> convertToTensors(const T1& outputs, const T2& shapes) {
  std::vector<vision_core::Tensor> tensors;
  tensors.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    tensors.emplace_back(outputs[i], shapes[i]);
  }
  return tensors;
}

} // namespace

void VisionApp::warmup_gpu(const cv::Mat &image) {
 try {
  for (int i = 0; i < 5; ++i) { // Warmup for 5 iterations
   // Use vision-core preprocessing
   const auto preprocessed = task->preprocess({image});

   // Pass preprocessed data directly to engine
   const auto [outputs, shapes] = engine->get_infer_results(preprocessed);

   auto tensors = convertToTensors(outputs, shapes);
   auto results = task->postprocess(image.size(), tensors);
   // Process results for warmup (no need to visualize)
   (void)results; // Suppress unused variable warning
  }
 } catch (const std::exception &e) {
  LOG(ERROR) << "Error: " << e.what();
  throw;
 }
}

void VisionApp::benchmark(const cv::Mat &image) {
 try {
  double total_time = 0.0;
  for (int i = 0; i < config.benchmark_iterations; ++i) {
   auto start = std::chrono::steady_clock::now();
   
   // Use vision-core preprocessing
   const auto preprocessed = task->preprocess({image});

   // Pass preprocessed data directly to engine
   const auto [outputs, shapes] = engine->get_infer_results(preprocessed);
   
   auto tensors = convertToTensors(outputs, shapes);
   auto results = task->postprocess(image.size(), tensors);
   
   // Process results for benchmark (no need to visualize)
   (void)results; // Suppress unused variable warning
   auto end = std::chrono::steady_clock::now();
   auto duration =
     std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
       .count();
   LOG(INFO) << "Iteration " << i << ": " << duration << "ms";
   total_time += duration;
  }
  double average_time = total_time / config.benchmark_iterations;
  LOG(INFO) << "Average inference time over " << config.benchmark_iterations
       << " iterations: " << average_time << "ms";
 } catch (const std::exception &e) {
  LOG(ERROR) << "Error: " << e.what();
  throw;
 }
}

void VisionApp::processImage(const std::string &source) {
 try {
  cv::Mat image = cv::imread(source);
  if (config.enable_warmup) {
   LOG(INFO) << "Warmup...";
   warmup_gpu(image); // Warmup before inference
  }
  auto start = std::chrono::steady_clock::now();
  
  // Get input dimensions from model metadata for logging purposes
  const auto inference_metadata = engine->get_inference_metadata();
  const auto &first_input = inference_metadata.getInputs()[0];
  auto [batch, channels, height, width] = extractInputDims(first_input.shape);

  LOG(INFO) << "Model input shape: " << batch << "x" << channels << "x"
       << height << "x" << width;
  LOG(INFO) << "Image dimensions: " << image.rows << "x" << image.cols << "x"
       << image.channels();

  // Use vision-core preprocessing
  const auto preprocessed = task->preprocess({image});

  // Pass preprocessed data directly to engine
  const auto [outputs, shapes] = engine->get_infer_results(preprocessed);
  
  auto tensors = convertToTensors(outputs, shapes);
  auto results = task->postprocess(image.size(), tensors);
  auto end = std::chrono::steady_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
  LOG(INFO) << "Inference time: " << duration << " ms";
   
  // Process results based on task type
  processResults(results, image);
  std::filesystem::create_directories("data/output");
  std::string processed_path = "data/output/processed.png";
  if (!cv::imwrite(processed_path, image)) {
   const std::string fallback_path = "/tmp/vision-inference-processed.png";
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
  if (config.enable_benchmark) {
   benchmark(image); // Benchmark
  }
 } catch (const std::exception &e) {
  LOG(ERROR) << "Error: " << e.what();
  throw;
 }
}

void VisionApp::processVideo(const std::string &source) {
 try {
  std::unique_ptr<VideoCaptureInterface> videoInterface =
    createVideoInterface();

  if (!videoInterface->initialize(source)) {
   throw std::runtime_error(
     "Failed to initialize video capture for input: " + source);
  }

  cv::Mat frame;
  while (videoInterface->readFrame(frame)) {
   auto start = std::chrono::steady_clock::now();
   
   // Use vision-core preprocessing
   const auto preprocessed = task->preprocess({frame});

   // Pass preprocessed data directly to engine
   const auto [outputs, shapes] = engine->get_infer_results(preprocessed);
   
   auto tensors = convertToTensors(outputs, shapes);
   auto results = task->postprocess(frame.size(), tensors);
   auto end = std::chrono::steady_clock::now();
   auto duration =
     std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
       .count();
   double fps = 1000.0 / duration;
   std::string fpsText = "FPS: " + std::to_string(fps);
   cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
         1, cv::Scalar(0, 255, 0), 2);
    
   // Process results based on task type
   processResults(results, frame);

   cv::imshow("opencv feed", frame);
   char key = cv::waitKey(1);
   if (key == 27 || key == 'q') {
    LOG(INFO) << "Exit requested";
    break;
   }
  }

  videoInterface->release();
 } catch (const std::exception &e) {
  LOG(ERROR) << "Error: " << e.what();
  throw;
 }
}

int VisionApp::getRequiredFrameCount() const {
 // Use CLI override if provided, otherwise get from task
 if (config.num_frames > 0) {
  return config.num_frames;
 }
 return task ? task->getRequiredFrames() : 1;
}

void VisionApp::processVideoClassification(const std::string &source) {
 try {
  std::unique_ptr<VideoCaptureInterface> videoInterface =
    createVideoInterface();

  if (!videoInterface->initialize(source)) {
   throw std::runtime_error(
     "Failed to initialize video capture for input: " + source);
  }

  const int requiredFrames = getRequiredFrameCount();
  LOG(INFO) << "Video classification mode: accumulating " << requiredFrames << " frames";

  cv::Mat frame;
  frameBuffer.clear();
  frameBuffer.reserve(requiredFrames);

  while (videoInterface->readFrame(frame)) {
   // Accumulate frames
   frameBuffer.push_back(frame.clone());

   // Process when we have enough frames
   if (static_cast<int>(frameBuffer.size()) >= requiredFrames) {
    auto start = std::chrono::steady_clock::now();

    // Use vision-core preprocessing with accumulated frames
    const auto preprocessed = task->preprocess(frameBuffer);

    // Pass preprocessed data directly to engine
    const auto [outputs, shapes] = engine->get_infer_results(preprocessed);

    auto tensors = convertToTensors(outputs, shapes);
    auto results = task->postprocess(frame.size(), tensors);
    auto end = std::chrono::steady_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
    double fps = 1000.0 / duration;
    std::string fpsText = "FPS: " + std::to_string(fps);

    // Display on the most recent frame
    cv::Mat displayFrame = frameBuffer.back().clone();
    cv::putText(displayFrame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
          1, cv::Scalar(0, 255, 0), 2);

    // Process results based on task type
    processResults(results, displayFrame);

    cv::imshow("opencv feed", displayFrame);

    // Use sliding window: remove oldest frame, keep the rest
    frameBuffer.erase(frameBuffer.begin());
   }

   char key = cv::waitKey(1);
   if (key == 27 || key == 'q') {
    LOG(INFO) << "Exit requested";
    break;
   }
  }

  videoInterface->release();
 } catch (const std::exception &e) {
  LOG(ERROR) << "Error: " << e.what();
  throw;
 }
}

void VisionApp::processOpticalFlow() {
 // Process optical flow with multiple input tensors
 // Now supported with updated neuriplo library
  
 LOG(INFO) << "Processing optical flow for image pairs";
  
 // Process pairs of images
 for(size_t i = 0; i < config.sources.size() - 1; i++) {
  std::vector<std::string> flowInputs = {config.sources[i], config.sources[i+1]};
   
  // Load images
  std::vector<cv::Mat> images;
  for (const auto& name : flowInputs) {
   cv::Mat img = cv::imread(name);
   if (img.empty()) {
    LOG(ERROR) << "Could not open or read the image: " << name;
    continue;
   }
   images.push_back(img);
  }
   
  if (images.size() != 2) continue;

  auto start = std::chrono::steady_clock::now();
   
  // Use vision-core preprocessing
  const auto preprocessed = task->preprocess(images);
   
  // Run inference with multiple input tensors directly from preprocessed data
  auto [infer_results, infer_shapes] = engine->get_infer_results(preprocessed);
   
  // Use vision-core postprocessing
  auto tensors = convertToTensors(infer_results, infer_shapes);
  auto predictions = task->postprocess(cv::Size(images[0].cols, images[0].rows), 
                    tensors);
   
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  LOG(INFO) << "Infer time for " << images.size() << " images: " << diff << " ms";
   
  // Visualization for optical flow
  cv::Mat& image = images[0];
  for (const auto& prediction : predictions) {
   if (std::holds_alternative<vision_core::OpticalFlow>(prediction)) {
    vision_core::OpticalFlow flow = std::get<vision_core::OpticalFlow>(prediction);
    flow.flow.copyTo(image);
   }
  }
   
  // Save result
  std::string sourceDir = flowInputs[0].substr(0, flowInputs[0].find_last_of("/\\"));
  std::string outputDir = sourceDir + "/output";
  std::filesystem::create_directories(outputDir);
  std::string processedFrameFilename = outputDir + "/processed_frame_optical_flow.jpg";
  LOG(INFO) << "Saving frame to: " << processedFrameFilename;
  cv::imwrite(processedFrameFilename, image);
 }
}

std::tuple<int, int, int, int>
VisionApp::extractInputDims(const std::vector<int64_t> &shape) {
 if (shape.size() == 4) {
  return {static_cast<int>(shape[0]), static_cast<int>(shape[1]),
      static_cast<int>(shape[2]), static_cast<int>(shape[3])};
 } else if (shape.size() == 3) {
  // Assume CHW with batch size 1
  return {1, static_cast<int>(shape[0]), static_cast<int>(shape[1]),
      static_cast<int>(shape[2])};
 } else {
  throw std::runtime_error(
    "Invalid input shape: expected 3D (CHW) or 4D (NCHW) tensor");
 }
}
