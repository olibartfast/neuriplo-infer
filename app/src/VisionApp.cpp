#include "VisionApp.hpp"
#include <algorithm>
#include <filesystem>
#include <chrono>

VisionApp::VisionApp(const AppConfig &config)
    : config(config) {
  try {
    setupLogging();

    LOG(INFO) << "Sources: ";
    for (const auto& src : config.sources) {
        LOG(INFO) << "  " << src;
    }
    LOG(INFO) << "Weights " << config.weights;
    LOG(INFO) << "Labels file " << config.labelsPath;
    LOG(INFO) << "Detector type " << config.detectorType;

    classes = readLabelNames(config.labelsPath);

    LOG(INFO) << "CPU info " << getCPUInfo();
    const auto gpuInfo = getGPUModel();
    LOG(INFO) << "GPU info: " << gpuInfo;
    const auto use_gpu = config.use_gpu && hasNvidiaGPU();
    engine = setup_inference_engine(config.weights, use_gpu, config.batch_size,
                                    config.input_sizes);
    if (!engine) {
      throw std::runtime_error("Can't setup an inference engine for " +
                               config.weights);
    }

    const auto inference_metadata = engine->get_inference_metadata();
    vision_core::ModelInfo model_info;
    for (const auto &input : inference_metadata.getInputs()) {
      auto shape = input.shape;
      // Normalize 3D shape (C,H,W) to 4D (1,C,H,W) to satisfy vision-core
      if (shape.size() == 3) {
        shape.insert(shape.begin(), 1); // Add batch dim
      }
      model_info.addInput(input.name, shape, input.batch_size);
    }
    for (const auto &output : inference_metadata.getOutputs()) {
      model_info.addOutput(output.name, output.shape, output.batch_size);
    }

    // Set input format for vision-core based on shape
    if (!model_info.input_formats.empty() && !model_info.input_shapes.empty() &&
        !model_info.input_shapes[0].empty()) {
      const auto &shape = model_info.input_shapes[0];
      if (shape.size() == 4) {
        // Heuristic: check if channels (1 or 3) are at index 1 (NCHW) or index
        // 3 (NHWC) NCHW: [Batch, Channels, Height, Width] NHWC: [Batch, Height,
        // Width, Channels]

        bool is_nchw = (shape[1] == 1 || shape[1] == 3);
        bool is_nhwc = (shape[3] == 1 || shape[3] == 3);

        if (is_nchw && !is_nhwc) {
          model_info.input_formats[0] = "FORMAT_NCHW";
        } else if (!is_nchw && is_nhwc) {
          model_info.input_formats[0] = "FORMAT_NHWC";
        } else {
          // Ambiguous or neither (e.g. 1x3x224x3 or non-standard).
          // Defaulting to NCHW as it's standard for ONNX/Torch vision models.
          // If height/width are significantly larger, we can use that to
          // disambiguate.
          if (shape[2] > 3 && shape[3] > 3) {
            model_info.input_formats[0] = "FORMAT_NCHW";
          } else if (shape[1] > 3 && shape[2] > 3) {
            model_info.input_formats[0] = "FORMAT_NHWC";
          } else {
            model_info.input_formats[0] = "FORMAT_NCHW";
          }
        }
      } else if (shape.size() == 3) {
        // Handle 3D shape (C, H, W) or (H, W, C)
        // Assume CHW if first dim is small (channels)
        if (shape[0] == 1 || shape[0] == 3) {
          model_info.input_formats[0] = "FORMAT_NCHW";
        } else {
          // Fallback to NCHW as safe default
          model_info.input_formats[0] = "FORMAT_NCHW";
        }
      }
    }

    // Set input type (float32)
    if (!model_info.input_types.empty()) {
      model_info.input_types[0] = CV_32F;
    }

    // Use exact model type from docs/TablePage.md
    // Valid types: yolov4, yolo, yolonas, rtdetr, rtdetrul, rfdetr
    LOG(INFO) << "Using vision-core model type: " << config.detectorType;

    task = vision_core::TaskFactory::createTaskInstance(config.detectorType,
                                                        model_info);
    if (!task) {
      throw std::runtime_error("Can't setup a task for " + config.detectorType);
    }

  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    throw;
  }
}

void VisionApp::run() {
  try {
    // Check if we have image files
    bool hasImages = false;
    for (const auto& src : config.sources) {
      if (src.find(".jpg") != std::string::npos || 
          src.find(".png") != std::string::npos) {
        hasImages = true;
        break;
      }
    }
    
    if (hasImages) {
      if (config.sources.size() == 1) {
        processImage(config.sources[0]);
      } else if (config.sources.size() >= 2 && 
                 getTaskType(config.detectorType) == vision_core::TaskType::OpticalFlow) {
        processOpticalFlow();
      } else {
        LOG(ERROR) << "Multiple image sources only supported for optical flow";
        throw std::runtime_error("Multiple image sources only supported for optical flow");
      }
    } else {
      if (config.sources.size() != 1) {
        LOG(ERROR) << "Video processing requires single source";
        throw std::runtime_error("Video processing requires single source");
      }
      processVideo(config.sources[0]);
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    throw;
  }
}

void VisionApp::warmup_gpu(const cv::Mat &image) {
  try {
    for (int i = 0; i < 5; ++i) { // Warmup for 5 iterations
      // Use vision-core preprocessing
      const auto preprocessed = task->preprocess({image});

      // Get input dimensions from model metadata
      const auto inference_metadata = engine->get_inference_metadata();
      const auto &first_input = inference_metadata.getInputs()[0];

      // Extract actual dimensions from model shape
      auto [batch, channels, height, width] =
          extractInputDims(first_input.shape);

      // Create cv::Mat from preprocessed data with correct shape
      cv::Mat input_blob(std::vector<int>{batch, channels, height, width},
                         CV_32F);
      std::memcpy(input_blob.data, preprocessed[0].data(),
                  preprocessed[0].size());
      const auto [outputs, shapes] = engine->get_infer_results(input_blob);
      auto results = task->postprocess(image.size(), outputs, shapes);
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

      // Get input dimensions from model metadata
      const auto inference_metadata = engine->get_inference_metadata();
      const auto &first_input = inference_metadata.getInputs()[0];

      // Extract actual dimensions from model shape
      auto [batch, channels, height, width] =
          extractInputDims(first_input.shape);

      // Create cv::Mat from preprocessed data with correct shape
      cv::Mat input_blob(std::vector<int>{batch, channels, height, width},
                         CV_32F);
      std::memcpy(input_blob.data, preprocessed[0].data(),
                  preprocessed[0].size());
      const auto [outputs, shapes] = engine->get_infer_results(input_blob);
      auto results = task->postprocess(image.size(), outputs, shapes);
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
              << " iterations:  " << average_time << "ms";
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    throw;
  }
}

void VisionApp::setupLogging(const std::string &log_folder) {
  try {
    // Create logs folder if it doesn't exist
    if (!std::filesystem::exists(log_folder)) {
      std::filesystem::create_directory(log_folder);
    } else {
      // Clean old logs
      std::filesystem::directory_iterator end_itr;
      for (std::filesystem::directory_iterator itr(log_folder); itr != end_itr;
           ++itr) {
        std::filesystem::remove(itr->path());
      }
    }

    // Initialize Google Logging
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

void VisionApp::processImage(const std::string &source) {
  try {
    cv::Mat image = cv::imread(source);
    if (config.enable_warmup) {
      LOG(INFO) << "Warmup...";
      warmup_gpu(image); // Warmup before inference
    }
    auto start = std::chrono::steady_clock::now();
    // Get input dimensions from model metadata
    const auto inference_metadata = engine->get_inference_metadata();
    const auto &first_input = inference_metadata.getInputs()[0];

    // Extract actual dimensions from model shape
    auto [batch, channels, height, width] = extractInputDims(first_input.shape);

    LOG(INFO) << "Model input shape: " << batch << "x" << channels << "x"
              << height << "x" << width;
    LOG(INFO) << "Image dimensions: " << image.rows << "x" << image.cols << "x"
              << image.channels();

    // Use vision-core preprocessing
    const auto preprocessed = task->preprocess({image});

    // Create cv::Mat from preprocessed data with correct shape
    cv::Mat input_blob(std::vector<int>{batch, channels, height, width},
                       CV_32F);
    std::memcpy(input_blob.data, preprocessed[0].data(),
                preprocessed[0].size());
    const auto [outputs, shapes] = engine->get_infer_results(input_blob);
    auto results = task->postprocess(image.size(), outputs, shapes);
    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    LOG(INFO) << "Inference time: " << duration << " ms";
    
    // Process results based on task type
    processResults(results, image);
    cv::imwrite("data/processed.png", image);
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

      // Get input dimensions from model metadata
      const auto inference_metadata = engine->get_inference_metadata();
      const auto &first_input = inference_metadata.getInputs()[0];

      // Extract actual dimensions from model shape
      auto [batch, channels, height, width] =
          extractInputDims(first_input.shape);

      // Create cv::Mat from preprocessed data with correct shape
      cv::Mat input_blob(std::vector<int>{batch, channels, height, width},
                         CV_32F);
      std::memcpy(input_blob.data, preprocessed[0].data(),
                  preprocessed[0].size());
      const auto [outputs, shapes] = engine->get_infer_results(input_blob);
      auto results = task->postprocess(frame.size(), outputs, shapes);
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
    
    // Get input dimensions from model metadata
    const auto inference_metadata = engine->get_inference_metadata();
    const auto inputs = inference_metadata.getInputs();
    
    // Use vision-core preprocessing
    const auto preprocessed = task->preprocess(images);
    
    // Create input tensors for both frames
    std::vector<cv::Mat> input_tensors;
    
    for (size_t i = 0; i < std::min(inputs.size(), preprocessed.size()); i++) {
      // Use provided input sizes if available, otherwise extract from model metadata
      std::vector<int> dims;
      if (i < config.input_sizes.size() && !config.input_sizes[i].empty()) {
        // Use provided input sizes: convert int64_t to int and add batch dimension
        dims = {1}; // batch size
        for (auto dim : config.input_sizes[i]) {
          dims.push_back(static_cast<int>(dim));
        }
      } else {
        // Fallback to model metadata
        auto [batch, channels, height, width] = extractInputDims(inputs[i].shape);
        dims = {batch, channels, height, width};
      }
      
      LOG(INFO) << "Input " << i << " dimensions: " << dims[0] << "x" << dims[1] << "x" << dims[2] << "x" << dims[3];
      
      // Calculate expected total size
      size_t expected_size = 1;
      for (int dim : dims) {
        expected_size *= dim;
      }
      expected_size *= sizeof(float); // CV_32F uses float
      
      LOG(INFO) << "Preprocessed data size: " << preprocessed[i].size() << " bytes, expected: " << expected_size << " bytes";
      
      if (preprocessed[i].size() != expected_size) {
        LOG(WARNING) << "Data size mismatch! Using available data size for tensor creation";
        // Calculate what dimensions we can actually support
        size_t float_count = preprocessed[i].size() / sizeof(float);
        LOG(INFO) << "Available float elements: " << float_count;
      }
      
      // Create cv::Mat from preprocessed data with correct shape
      cv::Mat input_blob(dims, CV_32F);
      size_t copy_size = std::min(preprocessed[i].size(), static_cast<size_t>(input_blob.total() * input_blob.elemSize()));
      std::memcpy(input_blob.data, preprocessed[i].data(), copy_size);
      input_tensors.push_back(input_blob);
    }
    
    // Run inference with multiple input tensors
    auto [infer_results, infer_shapes] = engine->get_infer_results(input_tensors);
    
    // Use vision-core postprocessing
    auto predictions = task->postprocess(cv::Size(images[0].cols, images[0].rows), 
                                        infer_results, infer_shapes);
    
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

vision_core::TaskType VisionApp::getTaskType(const std::string& model_type) {
  std::string normalized = model_type;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
  
  if (normalized == "torchvisionclassifier" || normalized == "tensorflowclassifier" || 
      normalized == "vitclassifier" || normalized == "timesformer") {
    return vision_core::TaskType::Classification;
  } else if (normalized.find("seg") != std::string::npos || normalized == "yoloseg") {
    return vision_core::TaskType::InstanceSegmentation;
  } else if (normalized == "raft") {
    return vision_core::TaskType::OpticalFlow;
  } else {
    return vision_core::TaskType::Detection; // Default for YOLO, RTDETR, etc.
  }
}

void VisionApp::processResults(const std::vector<vision_core::Result> &results, cv::Mat &image) {
  auto task_type = getTaskType(config.detectorType);
  
  switch (task_type) {
    case vision_core::TaskType::Detection: {
      for (const auto &result : results) {
        if (std::holds_alternative<vision_core::Detection>(result)) {
          const auto &detection = std::get<vision_core::Detection>(result);
          cv::rectangle(image, detection.bbox, cv::Scalar(255, 0, 0), 3);
          draw_label(image, classes[static_cast<int>(detection.class_id)],
                    detection.class_confidence, detection.bbox.x, detection.bbox.y);
        }
      }
      break;
    }
    case vision_core::TaskType::Classification: {
      std::string result_text = "Classification: ";
      for (const auto &result : results) {
        if (std::holds_alternative<vision_core::Classification>(result)) {
          const auto &classification = std::get<vision_core::Classification>(result);
          if (classification.class_id >= 0 && classification.class_id < classes.size()) {
            result_text += classes[static_cast<int>(classification.class_id)] + 
                          " (" + std::to_string(classification.class_confidence) + ")";
          }
        } else if (std::holds_alternative<vision_core::VideoClassification>(result)) {
          const auto &video_classification = std::get<vision_core::VideoClassification>(result);
          result_text += video_classification.action_label + 
                        " (" + std::to_string(video_classification.class_confidence) + ")";
        }
      }
      cv::putText(image, result_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                  1, cv::Scalar(0, 255, 255), 2);
      break;
    }
    case vision_core::TaskType::InstanceSegmentation: {
      for (const auto &result : results) {
        if (std::holds_alternative<vision_core::InstanceSegmentation>(result)) {
          const auto &segmentation = std::get<vision_core::InstanceSegmentation>(result);
          // Draw bounding box
          cv::rectangle(image, segmentation.bbox, cv::Scalar(255, 0, 0), 3);
          draw_label(image, classes[static_cast<int>(segmentation.class_id)],
                    segmentation.class_confidence, segmentation.bbox.x, segmentation.bbox.y);
          
          // Overlay mask if available
          if (!segmentation.mask.empty()) {
            cv::Mat colored_mask;
            cv::applyColorMap(segmentation.mask, colored_mask, cv::COLORMAP_JET);
            cv::addWeighted(image, 0.7, colored_mask, 0.3, 0, image);
          }
        }
      }
      break;
    }
    case vision_core::TaskType::OpticalFlow: {
      for (const auto &result : results) {
        if (std::holds_alternative<vision_core::OpticalFlow>(result)) {
          const auto &flow = std::get<vision_core::OpticalFlow>(result);
          if (!flow.flow.empty()) {
            // Replace the image with the flow visualization
            image = flow.flow.clone();
          }
          std::string flow_text = "Max displacement: " + std::to_string(flow.max_displacement);
          cv::putText(image, flow_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                      1, cv::Scalar(255, 255, 255), 2);
        }
      }
      break;
    }
  }
}