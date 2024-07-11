#include "VideoCaptureFactory.hpp"
#include "DetectorSetup.hpp"
#include "InferenceBackendSetup.hpp"
#include "utils.hpp"

static const std::string params = "{ help h   |   | print help message }"
      "{ type     |  yolov9 | yolov4, yolov5, yolov6, yolov7,yolov8, yolov9, rtdetr, rtdetrul}"
      "{ source s   |   | path to image or video source}"
      "{ labels lb  |  | path to class labels}"
      "{ config c   |   | optional model configuration file}"
      "{ weights w  |   | path to models weights}"
      "{ use_gpu   | false  | activate gpu support}"
      "{ min_confidence | 0.25   | optional min confidence}";


// Function to setup glog logging
void setupLogging(const std::string& log_folder = "./logs") {
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
    google::InitGoogleLogging("object_detection");  // Initialize glog
    google::SetLogDestination(google::GLOG_INFO, (log_folder + "/log_info_").c_str());  // Log to info log file
    google::SetLogDestination(google::GLOG_WARNING, (log_folder + "/log_warning_").c_str());  // Log to warning log file
    google::SetLogDestination(google::GLOG_ERROR, (log_folder + "/log_error_").c_str());  // Log to error log file
    google::SetStderrLogging(google::GLOG_INFO);  // Log to console

    FLAGS_logbufsecs = 0;  // Flush log every time
    FLAGS_max_log_size = 100;  // Maximum log file size in MB
    FLAGS_stop_logging_if_full_disk = true;  // Stop logging if disk is full
}



int main(int argc, char *argv[]) {
    setupLogging();

    // Use the logger for logging
    LOG(INFO) << "Initializing application";

    // Command line parser
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

    std::string source = parser.get<std::string>("source");
    if (source.empty()) {
        LOG(ERROR) << "Cannot open video stream";
        std::exit(1);
    }
    LOG(INFO) << "Source " << source;

    const bool use_gpu = parser.get<bool>("use_gpu");
    const std::string config = parser.get<std::string>("config");
    if (!config.empty() && !isFile(config)) {
        LOG(ERROR) << "Conf file " << config << " doesn't exist";
        std::exit(1);
    }

    const std::string weights = parser.get<std::string>("weights");
    if (!isFile(weights) && getFileExtension(config) != "xml") {
        LOG(ERROR) << "Weights file " << weights << " doesn't exist";
        std::exit(1);
    }
    LOG(INFO) << "Weights " << weights;

    const std::string labelsPath = parser.get<std::string>("labels");
    if (!isFile(labelsPath)) {
        LOG(ERROR) << "Labels file " << labelsPath << " doesn't exist";
        std::exit(1);
    }
    LOG(INFO) << "Labels file " << labelsPath;

    const std::string detectorType = parser.get<std::string>("type");
    LOG(INFO) << "Detector type " << detectorType;

    float confidenceThreshold = parser.get<float>("min_confidence");
    std::vector<std::string> classes = readLabelNames(labelsPath);
    LOG(INFO) << "Current path is " << std::filesystem::current_path().c_str();

    std::unique_ptr<Detector> detector = createDetector(detectorType);

    if (!detector) {
        LOG(ERROR) << "Can't setup a detector " << detectorType;
        std::exit(1);
    }

    std::unique_ptr<InferenceInterface> engine = setup_inference_engine(weights, config);
    if (!engine) {
        LOG(ERROR) << "Can't setup an inference engine for " << weights << " " << config;
        std::exit(1);
    }

    if (source.find(".jpg") != std::string::npos || source.find(".png") != std::string::npos) {
        cv::Mat image = cv::imread(source);
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
        return 0;
    }

    std::unique_ptr<VideoCaptureInterface> videoInterface = createVideoInterface();

    if (!videoInterface->initialize(source)) {
        LOG(ERROR) << "Failed to initialize video capture for input: " << source;
        return 1;
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
    return 0;
}
