#include "VideoCaptureFactory.hpp"
#include "DetectorFactory.hpp"
#include "Logger.hpp"
#include "utils.hpp"


static const std::string params = "{ help h   |   | print help message }"
      "{ type     |  yolov8 | yolov4, yolov5, yolov6, yolov7, tensorflow, rtdetr}"
      "{ source s   |   | path to image or video source}"
      "{ labels lb  |  | path to class labels}"
      "{ conf c   |   | optional model configuration file}"
      "{ weights w  |   | path to models weights}"
      "{ use_gpu   | false  | activate gpu support}"
      "{ min_confidence | 0.25   | optional min confidence}";


int main (int argc, char *argv[])
{
    initializeLogger();

    // Use the logger for logging
    logger->info("Initializing application");

    // Command line parser
    cv::CommandLineParser parser(argc, argv, params);
    parser.about("Detect objets from video or image input source");
    if (parser.has("help")){
        parser.printMessage();
        std::exit(1);  
    }
    std::string source = parser.get<std::string>("source");
    if (!parser.check())
    {
        parser.printErrors();
        std::exit(1);
    }
    if (source.empty()){
        logger->error("Can not open video stream" );
        std::exit(1);
    }

    const bool use_gpu = parser.get<bool>("use_gpu");

    const std::string weights = parser.get<std::string>("weights");
    if(!isFile(weights))
    {
         logger->error("weights file {} doesn't exist", weights);
         std::exit(1);
    }

    const std::string labelsPath = parser.get<std::string>("labels");
    if(!isFile(labelsPath))
    {
         logger->error("labels file {} doesn't exist", labelsPath);
         std::exit(1);
    }

    const std::string conf =  parser.get<std::string>("conf");
    if(!conf.empty() && !isFile(conf))
    {
         logger->error("conf file {} doesn't exist", conf);
         std::exit(1);
    }

    const std::string detectorType = parser.get<std::string>("type");
    float confidenceThreshold = parser.get<float>("min_confidence");
    std::vector<std::string> classes = readLabelNames(labelsPath); 
    logger->info("Current path is {}", std::filesystem::current_path().c_str()); 

    Detector::SetLogger(logger);
    std::unique_ptr<Detector> detector = createDetector(detectorType, use_gpu, labelsPath, weights, conf); 
    if(!detector)
    {
        logger->error("Detector creation fail!");
        std::exit(1);
    }

    if (source.find(".jpg") != std::string::npos || source.find(".png") != std::string::npos) 
    {
        cv::Mat image = cv::imread(source);
        auto start = std::chrono::steady_clock::now();
        std::vector<Detection> detections = detector->run_detection(image);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        logger->info("Inference time: {} ms", duration);
        for (const auto& d : detections) 
        {
            cv::rectangle(image, d.bbox, cv::Scalar(255, 0, 0), 3);
            draw_label(image, classes[d.label], d.score, d.bbox.x, d.bbox.y);
        }        
        cv::imwrite("data/processed.png", image);
        return 0;
    }

    std::unique_ptr<VideoCaptureInterface> videoInterface = createVideoInterface();

    if (!videoInterface->initialize(source)) {
        logger->error("Failed to initialize video capture for input: {}", source);
        return 1;
    }    

    cv::Mat frame;
    while ( videoInterface->readFrame(frame)) 
    {
        auto start = std::chrono::steady_clock::now();
        std::vector<Detection> detections = detector->run_detection(frame);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double fps = 1000.0 / duration;
        std::string fpsText = "FPS: " + std::to_string(fps);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        for (const auto& d : detections) {
            cv::rectangle(frame, d.bbox, cv::Scalar(255, 0, 0), 3);
            draw_label(frame, classes[d.label], d.score, d.bbox.x, d.bbox.y);
        }

        cv::imshow("opencv feed", frame);
        char key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            logger->info("Exit requested");
            break;
        }
    }
    
    videoInterface->release();
    return 0;  
}
