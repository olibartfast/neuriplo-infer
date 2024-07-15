#include "ObjectDetectionApp.hpp"

static const std::string params = "{ help h   |   | print help message }"
      "{ type     |  yolov10 | yolov4, yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, rtdetr, rtdetrul}"
      "{ source s   |   | path to image or video source}"
      "{ labels lb  |  | path to class labels}"
      "{ config c   |   | optional model configuration file}"
      "{ weights w  |   | path to models weights}"
      "{ use-gpu   | false  | activate gpu support}"
      "{ min_confidence | 0.25   | optional min confidence}"
      "{ warmup     | false  | enable GPU warmup}"
      "{ benchmark  | false  | enable benchmarking}"
      "{ iterations | 10     | number of iterations for benchmarking}";

AppConfig parseCommandLineArguments(int argc, char *argv[]) {
    cv::CommandLineParser parser(argc, argv, params);
    parser.about("Detect objects from video or image input source");

    if (parser.has("help")) {
        parser.printMessage();
        std::exit(1);
    }

    if (!parser.check()) {
        parser.printErrors();
        std::exit(1);
    }

    AppConfig config;
    config.source = parser.get<std::string>("source");
    if (config.source.empty()) {
        LOG(ERROR) << "Cannot open video stream";
        std::exit(1);
    }

    config.use_gpu = parser.get<bool>("use-gpu");
    config.enable_warmup = parser.get<bool>("warmup");
    config.enable_benchmark = parser.get<bool>("benchmark");
    config.benchmark_iterations = parser.get<int>("iterations");
    config.confidenceThreshold = parser.get<float>("min_confidence");
    config.config = parser.get<std::string>("config");
    if (!config.config.empty() && !isFile(config.config)) {
        LOG(ERROR) << "Conf file " << config.config << " doesn't exist";
        std::exit(1);
    }

    config.weights = parser.get<std::string>("weights");
    if (!isFile(config.weights) && getFileExtension(config.config) != "xml") {
        LOG(ERROR) << "Weights file " << config.weights << " doesn't exist";
        std::exit(1);
    }
    config.labelsPath = parser.get<std::string>("labels");
    if (!isFile(config.labelsPath)) {
        LOG(ERROR) << "Labels file " << config.labelsPath << " doesn't exist";
        std::exit(1);
    }
    config.detectorType = parser.get<std::string>("type");

    return config;
}

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
