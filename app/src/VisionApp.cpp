#include "VisionApp.hpp"
#include <filesystem>

VisionApp::VisionApp(const AppConfig &config)
  : config(config) {
 try {
  setupLogging();

  LOG(INFO) << "Sources: ";
  for (const auto& src : config.sources) {
    LOG(INFO) << " " << src;
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
  for (size_t i = 0; i < inference_metadata.getInputs().size(); i++) {
   const auto &input = inference_metadata.getInputs()[i];
   std::vector<int64_t> shape;
    
   // Use command line input sizes if provided, otherwise use model metadata
   if (i < config.input_sizes.size() && !config.input_sizes[i].empty()) {
    shape = {}; // batch size
    for (auto dim : config.input_sizes[i]) {
     shape.push_back(dim);
    }
    // Normalize 3D shape (C,H,W) to 4D (1,C,H,W) for CLI inputs
    if (shape.size() == 3) {
     shape.insert(shape.begin(), config.batch_size); 
    }
   } else {
    shape = input.shape;
    // Normalize 3D shape (C,H,W) to 4D (1,C,H,W) to satisfy vision-core
    if (shape.size() == 3) {
     shape.insert(shape.begin(), config.batch_size); // Add batch dim
    }
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
   // Use video classification processing for temporal models
   if (getTaskType(config.detectorType) == vision_core::TaskType::VideoClassification) {
    processVideoClassification(config.sources[0]);
   } else {
    processVideo(config.sources[0]);
   }
  }
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
