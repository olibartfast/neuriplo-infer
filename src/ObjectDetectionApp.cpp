#include "ObjectDetectionApp.hpp"

ObjectDetectionApp::ObjectDetectionApp(const AppConfig& config)
    : config(config) {
    setupLogging();

    LOG(INFO) << "Source " << config.source;
    LOG(INFO) << "Weights " << config.weights;
    LOG(INFO) << "Labels file " << config.labelsPath;
    LOG(INFO) << "Detector type " << config.detectorType;

    classes = readLabelNames(config.labelsPath);
    
    detector = createDetector(config.detectorType);
    if (!detector) {
        LOG(ERROR) << "Can't setup a detector " << config.detectorType;
        std::exit(1);
    }

    engine = setup_inference_engine(config.weights, config.config, config.use_gpu);
    if (!engine) {
        LOG(ERROR) << "Can't setup an inference engine for " << config.weights << " " << config.config;
        std::exit(1);
    }
}

void ObjectDetectionApp::run() {
    if (config.source.find(".jpg") != std::string::npos || config.source.find(".png") != std::string::npos) {
        processImage(config.source);
    } else {
        processVideo(config.source);
    }
}

void ObjectDetectionApp::warmup_gpu(const cv::Mat& image) {
    for (int i = 0; i < 5; ++i) { // Warmup for 5 iterations
        const auto input_blob = detector->preprocess_image(image);
        const auto[outputs, shapes] = engine->get_infer_results(input_blob);
        std::vector<Detection> detections = detector->postprocess(outputs, shapes, image.size());
    }
}

void ObjectDetectionApp::benchmark(const cv::Mat& image) {
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
}

void ObjectDetectionApp::setupLogging(const std::string& log_folder) {
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
}

void ObjectDetectionApp::processImage(const std::string& source) {
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
}

void ObjectDetectionApp::processVideo(const std::string& source) {
    std::unique_ptr<VideoCaptureInterface> videoInterface = createVideoInterface();

    if (!videoInterface->initialize(source)) {
        LOG(ERROR) << "Failed to initialize video capture for input: " << source;
        std::exit(1);
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
}
