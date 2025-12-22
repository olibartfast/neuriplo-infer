#include "ObjectDetectionApp.hpp"

ObjectDetectionApp::ObjectDetectionApp(const AppConfig &config)
    : config(config) {
  try {
    setupLogging();

    LOG(INFO) << "Source " << config.source;
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
      model_info.addInput(input.name, input.shape, input.batch_size);
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

void ObjectDetectionApp::run() {
  try {
    if (config.source.find(".jpg") != std::string::npos ||
        config.source.find(".png") != std::string::npos) {
      processImage(config.source);
    } else {
      processVideo(config.source);
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    throw;
  }
}

void ObjectDetectionApp::warmup_gpu(const cv::Mat &image) {
  try {
    for (int i = 0; i < 5; ++i) { // Warmup for 5 iterations
      // Use vision-core preprocessing
      const auto preprocessed = task->preprocess({image});

      // Get input dimensions from model metadata
      const auto inference_metadata = engine->get_inference_metadata();
      const auto &first_input = inference_metadata.getInputs()[0];

      if (first_input.shape.size() < 4) {
        throw std::runtime_error(
            "Invalid input shape: expected 4D tensor (NCHW)");
      }

      // Extract actual dimensions from model shape
      int batch = first_input.shape[0];
      int channels = first_input.shape[1];
      int height = first_input.shape[2];
      int width = first_input.shape[3];

      // Create cv::Mat from preprocessed data with correct shape
      cv::Mat input_blob(std::vector<int>{batch, channels, height, width},
                         CV_32F);
      std::memcpy(input_blob.data, preprocessed[0].data(),
                  preprocessed[0].size());
      const auto [outputs, shapes] = engine->get_infer_results(input_blob);
      auto results = task->postprocess(image.size(), outputs, shapes);
      // Results contain individual Detection variants, extract them
      std::vector<vision_core::Detection> detections;
      for (const auto &result : results) {
        if (std::holds_alternative<vision_core::Detection>(result)) {
          detections.push_back(std::get<vision_core::Detection>(result));
        }
      }
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    throw;
  }
}

void ObjectDetectionApp::benchmark(const cv::Mat &image) {
  try {
    double total_time = 0.0;
    for (int i = 0; i < config.benchmark_iterations; ++i) {
      auto start = std::chrono::steady_clock::now();
      // Use vision-core preprocessing
      const auto preprocessed = task->preprocess({image});

      // Get input dimensions from model metadata
      const auto inference_metadata = engine->get_inference_metadata();
      const auto &first_input = inference_metadata.getInputs()[0];

      if (first_input.shape.size() < 4) {
        throw std::runtime_error(
            "Invalid input shape: expected 4D tensor (NCHW)");
      }

      // Extract actual dimensions from model shape
      int batch = first_input.shape[0];
      int channels = first_input.shape[1];
      int height = first_input.shape[2];
      int width = first_input.shape[3];

      // Create cv::Mat from preprocessed data with correct shape
      cv::Mat input_blob(std::vector<int>{batch, channels, height, width},
                         CV_32F);
      std::memcpy(input_blob.data, preprocessed[0].data(),
                  preprocessed[0].size());
      const auto [outputs, shapes] = engine->get_infer_results(input_blob);
      auto results = task->postprocess(image.size(), outputs, shapes);
      // Results contain individual Detection variants, extract them
      std::vector<vision_core::Detection> detections;
      for (const auto &result : results) {
        if (std::holds_alternative<vision_core::Detection>(result)) {
          detections.push_back(std::get<vision_core::Detection>(result));
        }
      }
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

void ObjectDetectionApp::setupLogging(const std::string &log_folder) {
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

void ObjectDetectionApp::processImage(const std::string &source) {
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

    if (first_input.shape.size() < 4) {
      throw std::runtime_error(
          "Invalid input shape: expected 4D tensor (NCHW)");
    }

    // Extract actual dimensions from model shape
    int batch = first_input.shape[0];
    int channels = first_input.shape[1];
    int height = first_input.shape[2];
    int width = first_input.shape[3];

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
    // Results contain individual Detection variants, extract them
    std::vector<vision_core::Detection> detections;
    for (const auto &result : results) {
      if (std::holds_alternative<vision_core::Detection>(result)) {
        detections.push_back(std::get<vision_core::Detection>(result));
      }
    }
    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    LOG(INFO) << "Inference time: " << duration << " ms";
    for (const auto &d : detections) {
      cv::rectangle(image, d.bbox, cv::Scalar(255, 0, 0), 3);
      draw_label(image, classes[static_cast<int>(d.class_id)],
                 d.class_confidence, d.bbox.x, d.bbox.y);
    }
    cv::imwrite("data/processed.png", image);
    if (config.enable_benchmark) {
      benchmark(image); // Benchmark
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error: " << e.what();
    throw;
  }
}

void ObjectDetectionApp::processVideo(const std::string &source) {
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

      if (first_input.shape.size() < 4) {
        throw std::runtime_error(
            "Invalid input shape: expected 4D tensor (NCHW)");
      }

      // Extract actual dimensions from model shape
      int batch = first_input.shape[0];
      int channels = first_input.shape[1];
      int height = first_input.shape[2];
      int width = first_input.shape[3];

      // Create cv::Mat from preprocessed data with correct shape
      cv::Mat input_blob(std::vector<int>{batch, channels, height, width},
                         CV_32F);
      std::memcpy(input_blob.data, preprocessed[0].data(),
                  preprocessed[0].size());
      const auto [outputs, shapes] = engine->get_infer_results(input_blob);
      auto results = task->postprocess(frame.size(), outputs, shapes);
      // Results contain individual Detection variants, extract them
      std::vector<vision_core::Detection> detections;
      for (const auto &result : results) {
        if (std::holds_alternative<vision_core::Detection>(result)) {
          detections.push_back(std::get<vision_core::Detection>(result));
        }
      }
      auto end = std::chrono::steady_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      double fps = 1000.0 / duration;
      std::string fpsText = "FPS: " + std::to_string(fps);
      cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                  1, cv::Scalar(0, 255, 0), 2);
      for (const auto &d : detections) {
        cv::rectangle(frame, d.bbox, cv::Scalar(255, 0, 0), 3);
        draw_label(frame, classes[static_cast<int>(d.class_id)],
                   d.class_confidence, d.bbox.x, d.bbox.y);
      }

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