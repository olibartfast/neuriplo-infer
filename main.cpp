#include "GStreamerOpenCV.hpp"
#include "YoloV8.hpp"
#ifdef USE_OPENCV_DNN
#include "YoloV4.hpp"
#include "YoloVn.hpp"
#include "YoloNas.hpp"
#endif
#ifdef USE_TENSORFLOW
#include "TFDetectionAPI.hpp"
#endif


// Define a global logger variable
std::shared_ptr<spdlog::logger> logger;

void initializeLogger() {

    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    sinks.push_back( std::make_shared<spdlog::sinks::rotating_file_sink_mt>("output.log", 1024*1024*10, 3, true));
    logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));

    spdlog::register_logger(logger);
    logger->flush_on(spdlog::level::info);
}


static const std::string params = "{ help h   |   | print help message }"
      "{ type     |  yolov8 | yolov4, yolov5, yolov6, yolov7, tensorflow}"
      "{ link l   |   | capture video from ip camera}"
      "{ labels lb  |  | path to class labels}"
      "{ conf c   |   | model configuration file}"
      "{ weights w  |   | path to models weights}"
      "{ use_opencv_dnn   | true  | use opencv dnn module to do inference}"
      "{ use_gpu   | false  | activate gpu support}"
      "{ min_confidence | 0.25   | min confidence}";


bool isDirectory(const std::string& path) {
    std::filesystem::path fsPath(path);
    return std::filesystem::is_directory(fsPath);
}



void draw_label(cv::Mat& input_image, std::string label, int left, int top)
{
    
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
    const int THICKNESS = 1;
    cv::Scalar YELLOW = cv::Scalar(0, 255, 255);

    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = cv::max(top, label_size.height);
    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    cv::rectangle(input_image, tlc, brc, cv::Scalar(255, 0, 255), cv::FILLED);
    // Put the label on the black rectangle.
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

std::vector<std::string> readLabelNames(const std::string& fileName)
{
    if(!std::filesystem::exists(fileName)){
        std::cerr << "Wrong path to labels " <<  fileName << std::endl;
        exit(1);
    } 
    std::vector<std::string> classes;
    std::ifstream ifs(fileName.c_str());
    std::string line;
    while (getline(ifs, line))
    classes.push_back(line);
    return classes;   
}


std::unique_ptr<Detector> createDetector(
    const std::string& detectorType,
    const std::string& labels,
    const std::string& weights,
    const std::string& modelConfiguration = "")
 {
    if(detectorType.find("yolov8") != std::string::npos)  
    {
        return std::make_unique<YoloV8>(weights);
    }       
#ifdef USE_OPENCV_DNN
    else if(detectorType.find("yolov4") != std::string::npos)
    {
        if(modelConfiguration.empty() || !std::filesystem::exists(modelConfiguration))
        {
            std::cerr << "YoloV4 needs a configuration file" << std::endl;
            return nullptr;
        }    
        return std::make_unique<YoloV4>(modelConfiguration, weights);
    }   
    else if(detectorType.find("yolov5") != std::string::npos || 
        detectorType.find("yolov6") != std::string::npos  ||
        detectorType.find("yolov7") != std::string::npos)  
    {
        return std::make_unique<YoloVn>(weights);
    }
    else if(detectorType.find("yolonas") != std::string::npos)  
    {
        return std::make_unique<YoloNas>(weights);
    }     
#endif    
#ifdef USE_TENSORFLOW      
    else if(detectorType.find("tensorflow") != std::string::npos) 
    {
        if(isDirectory(weights))
            return std::make_unique<TFDetectionAPI>(weights);
        else
        {
            std::cerr << "In case of Tensorflow weights must be a path to the saved model folder" << std::endl;
            return nullptr;   
        }    
    }
#endif      
    else
    return nullptr;
}

int main (int argc, char *argv[])
{
    initializeLogger();

    // Use the logger for logging
    logger->info("Initializing application");

    // Command line parser
    cv::CommandLineParser parser(argc, argv, params);
    parser.about("Detect people from rtsp ip camera stream");
    if (parser.has("help")){
        parser.printMessage();
        std::exit(1);  
    }
    std::string link = parser.get<std::string>("link");
    if (!parser.check())
    {
        parser.printErrors();
        std::exit(1);
    }
    if (link.empty()){
        logger->error("Can not open video stream" );
        std::exit(1);
    }

    const bool use_opencv_dnn = parser.get<bool>("use_opencv_dnn");
    const bool use_gpu = parser.get<bool>("use_gpu");

    std::unique_ptr<GStreamerOpenCV> gstocv = std::make_unique<GStreamerOpenCV>();
    gstocv->initGstLibrary(argc, argv);
    gstocv->runPipeline(link);
    gstocv->checkError();
    gstocv->getSink();
    gstocv->setBus();
    gstocv->setState(GST_STATE_PLAYING);
  
    const std::string weights = parser.get<std::string>("weights");
    const std::string labelsPath = parser.get<std::string>("labels");
    const std::string detectorType = parser.get<std::string>("type");
    const std::string conf =  parser.get<std::string>("conf");
    float confidenceThreshold = parser.get<float>("min_confidence");
    std::vector<std::string> classes = readLabelNames(labelsPath); 
    logger->info("Current path is {}", std::filesystem::current_path().c_str()); 

    Detector::SetLogger(logger);
    std::unique_ptr<Detector> detector = createDetector(detectorType, labelsPath, weights, conf); 
    if(!detector)
    {
        std::cerr << "Detector creation fail!" << std::endl;
        std::exit(1);
    }

    while (true) 
    {
        gstocv->setMainLoopEvent(false);
        cv::Mat frame = gstocv->getFrame().clone();
        if (!frame.empty())
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
                draw_label(frame, classes[d.label], d.bbox.x, d.bbox.y);
            }

            cv::imshow("opencv feed", frame);
            char key = cv::waitKey(1);
            if (key == 27 || key == 'q') {
                logger->info("Exit requested");
                break;
            }
            if (GStreamerOpenCV::isEndOfStream()) {
                logger->info("Video stream has finished");
                break;
            }
    
        }
    }
    
    gstocv->setState(GST_STATE_NULL);
    return 0;  
}
