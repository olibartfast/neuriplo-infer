#include "ObjectDetectionApp.hpp"

ObjectDetectionApp::ObjectDetectionApp(const AppConfig& config)
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
        engine = setup_inference_engine(config.weights, use_gpu, config.batch_size, config.input_sizes);
        if (!engine) {
            throw std::runtime_error("Can't setup an inference engine for " + config.weights);
        }

        const auto model_info = engine->get_model_info();

        detector =  DetectorSetup::createDetector(config.detectorType, model_info);
        if (!detector) {
            throw std::runtime_error("Can't setup a detector " + config.detectorType);
        }

    } catch (const std::exception& e) {
        LOG(ERROR) << "Error: " << e.what();
        throw;
    }
}

void ObjectDetectionApp::run() {
    try {
        if (config.source.find(".jpg") != std::string::npos || config.source.find(".png") != std::string::npos) {
            processImage(config.source);
        } else {
            processVideo(config.source);
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error: " << e.what();
        throw;
    }
}

void ObjectDetectionApp::warmup_gpu(const cv::Mat& image) {
    try {
        for (int i = 0; i < 5; ++i) { // Warmup for 5 iterations
            const auto input_blob = detector->preprocess_image(image);
            const auto[outputs, shapes] = engine->get_infer_results(input_blob);
            std::vector<Detection> detections = detector->postprocess(outputs, shapes, image.size());
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error: " << e.what();
        throw;
    }
}

void ObjectDetectionApp::benchmark(const cv::Mat& image) {
    try {
        double total_time = 0.0;
        for (int i = 0; i < config.benchmark_iterations; ++i) {
            auto start = std::chrono::steady_clock::now();
            const auto input_blob = detector->preprocess_image(image);
            const auto[outputs, shapes] = engine->get_infer_results(input_blob);
            std::vector<Detection> detections = detector->postprocess(outputs, shapes, image.size());
            auto end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            LOG(INFO) << "Iteration " << i << ": " << duration << "ms";
            total_time += duration;
        }
        double average_time = total_time / config.benchmark_iterations;
        LOG(INFO) << "Average inference time over " << config.benchmark_iterations << " iterations:  "<< average_time << "ms";
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error: " << e.what();
        throw;
    }
}

void ObjectDetectionApp::setupLogging(const std::string& log_folder) {
    try {
        // Create logs folder if it doesn't exist
        if (!std::filesystem::exists(log_folder)) {
            std::filesystem::create_directory(log_folder);
        } else {
            // Clean old logs
            std::filesystem::directory_iterator end_itr;
            for (std::filesystem::directory_iterator itr(log_folder); itr != end_itr; ++itr) {
                std::filesystem::remove(itr->path());
            }
        }

        // Initialize Google Logging
        google::InitGoogleLogging("object_detection");
        google::SetLogDestination(google::GLOG_INFO, (log_folder + "/log_info_").c_str());
        google::SetLogDestination(google::GLOG_WARNING, (log_folder + "/log_warning_").c_str());
        google::SetLogDestination(google::GLOG_ERROR, (log_folder + "/log_error_").c_str());
        google::SetStderrLogging(google::GLOG_INFO);

        FLAGS_logbufsecs = 0;
        FLAGS_max_log_size = 100;
        FLAGS_stop_logging_if_full_disk = true;
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error: " << e.what();
        throw;
    }
}

void ObjectDetectionApp::processImage(const std::string& source) {
    try {
        cv::Mat image = cv::imread(source);
        if (config.enable_warmup) {
            LOG(INFO) << "Warmup...";
            warmup_gpu(image); // Warmup before inference
        }
        auto start = std::chrono::steady_clock::now();
        const auto input_blob = detector->preprocess_image(image);
        const auto [outputs, shapes] = engine->get_infer_results(input_blob);
        std::vector<Detection> detections = detector->postprocess(outputs, shapes, image.size());
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        LOG(INFO) << "Inference time: " << duration << " ms";
        for (const auto &d : detections) {
            cv::rectangle(image, d.bbox, cv::Scalar(255, 0, 0), 3);
            draw_label(image, classes[d.label], d.score, d.bbox.x, d.bbox.y);
        }
        cv::imwrite("data/processed.png", image);
        if (config.enable_benchmark) {
            benchmark(image); // Benchmark
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error: " << e.what();
        throw;
    }
}

void ObjectDetectionApp::processVideo(const std::string& source) {
    try {
        std::unique_ptr<VideoCaptureInterface> videoInterface = createVideoInterface();

        if (!videoInterface->initialize(source)) {
            throw std::runtime_error("Failed to initialize video capture for input: " + source);
        }

        cv::Mat frame;
        while (videoInterface->readFrame(frame)) {
            auto start = std::chrono::steady_clock::now();
            const auto input_blob = detector->preprocess_image(frame);
            const auto [outputs, shapes] = engine->get_infer_results(input_blob);
            std::vector<Detection> detections = detector->postprocess(outputs, shapes, frame.size());
            auto end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            double fps = 1000.0 / duration;
            std::string fpsText = "FPS: " + std::to_string(fps);
            cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            for (const auto &d : detections) {
                cv::rectangle(frame, d.bbox, cv::Scalar(255, 0, 0), 3);
                draw_label(frame, classes[d.label], d.score, d.bbox.x, d.bbox.y);
            }

            cv::imshow("opencv feed", frame);
            char key = cv::waitKey(1);
            if (key == 27 || key == 'q') {
                LOG(INFO) << "Exit requested";
                break;
            }
        }

        videoInterface->release();
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error: " << e.what();
        throw;
    }
}