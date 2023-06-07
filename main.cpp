#include "GStreamerOpenCV.hpp"
#include "Detector.hpp"
#include "HogSvmDetector.hpp"
#include "MobileNetSSD.hpp"
#include "YoloV4.hpp"
#include "YoloV5.hpp"
#include "YoloV8.hpp"
#ifdef USE_TENSORFLOW
#include "TFDetectionAPI.hpp"
#endif

static const std::string params = "{ help h   |   | print help message }"
      "{ type     |  yolov8x | mobilenet, svm, yolov4-tiny, yolov4, yolov5s, yolov5x, tensorflow}"
      "{ link l   |   | capture video from ip camera}"
      "{ labels lb   |  labels | path to class labels file}"
      "{ model_path mp   |  ../models | path to models}"
      "{ min_confidence | 0.25   | min confidence}";


void draw_label(cv::Mat& input_image, std::string label, int left, int top)
{
    
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
    const int THICKNESS = 1;
    cv::Scalar YELLOW = Scalar(0, 255, 255);

    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = cv::max(top, label_size.height);
    // Top left corner.
    Point tlc = cv::Point(left, top);
    // Bottom right corner.
    Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    cv::rectangle(input_image, tlc, brc, cv::Scalar(255, 0, 255), FILLED);
    // Put the label on the black rectangle.
    cv::putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
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

auto modelSetup(const std::string& modelPath, const std::string& configName, const std::string& weigthName){
    const auto modelConfiguration = modelPath + "/" + configName;
    const auto modelBinary = modelPath + "/" + weigthName;
    if(!std::filesystem::exists(modelConfiguration) || !std::filesystem::exists(modelBinary))
    {
        std::cerr << "Wrong path to model " << modelBinary << std::endl;
        exit(1);
    }
    return std::make_tuple(modelConfiguration, modelBinary); 
}

std::unique_ptr<Detector> createDetector(
    const std::string& detectorType,
    const std::string& labelsPath,
    const std::string& modelPath){
    std::vector<std::string> classes; 
    std::string modelConfiguration; 
    std::string modelBinary;  
    if(detectorType == "svm"){
        return std::make_unique<HogSvmDetector>();
    }
    else if(detectorType == "mobilenet")
    {
        classes = readLabelNames(labelsPath + "/" + "voc.names");      
        auto[modelConfiguration, modelBinary] = modelSetup(modelPath, "MobileNetSSD_deploy.prototxt",  "MobileNetSSD_deploy.caffemodel");
        return std::make_unique<MobileNetSSD>(classes, modelConfiguration, modelBinary);
    }
    else if(detectorType.find("yolov4") != std::string::npos)
    {
        std::string modelConfiguration, modelBinary;
        classes = readLabelNames(labelsPath + "/" + "coco.names"); 
        std::tie(modelConfiguration, modelBinary) = modelSetup(modelPath, detectorType + ".cfg",  detectorType + ".weights");  
        return std::make_unique<YoloV4>(classes, modelConfiguration, modelBinary);
    }   
    else if(detectorType.find("yolov5") != std::string::npos)  
    {
        std::string modelBinary;
        classes = readLabelNames(labelsPath + "/" + "coco.names"); 
        std::tie(modelConfiguration, modelBinary) = modelSetup(modelPath, "",  detectorType + ".onnx");    
        return std::make_unique<YoloV5>(classes, "", modelBinary);
    }
    else if(detectorType.find("yolov8") != std::string::npos)  
    {
        std::string modelBinary;
        classes = readLabelNames(labelsPath + "/" + "coco.names"); 
        std::tie(modelConfiguration, modelBinary) = modelSetup(modelPath, "",  detectorType + ".onnx");    
        return std::make_unique<YoloV8>(classes, "", modelBinary);
    }    
#ifdef USE_TENSORFLOW      
    else if(detectorType.find("tensorflow") != std::string::npos) 
    {
        return std::make_unique<TFDetectionAPI>(modelPath);
    }
#endif      
    return nullptr;
}

std::string getPipelineCommand(const std::string& link){
    if (link.find("rtsp") != std::string::npos)
        return "rtspsrc location=" + link + " ! decodebin ! appsink name=autovideosink";
    else
        return "filesrc location=" + link + " ! decodebin ! appsink name=autovideosink";    

}

int main (int argc, char *argv[])
{
    // Command line parser
    cv::CommandLineParser parser(argc, argv, params);
    parser.about("Detect people from rtsp ip camera stream");
    if (parser.has("help")){
      parser.printMessage();
      return 0;  
    }
    std::string link = parser.get<std::string>("link");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    if (link.empty()){
        std::cout << "Can not open video stream" << std::endl;
        return 2;
    }

    std::unique_ptr<GStreamerOpenCV> gstocv = std::make_unique<GStreamerOpenCV>();
    gstocv->init_gst_library(argc, argv);
    std::string pipeline_cmd = getPipelineCommand(link);
    std::cout << pipeline_cmd.c_str() << std::endl;
    gstocv->run_pipeline(pipeline_cmd);
    gstocv->check_error();
    gstocv->get_sink();
    gstocv->set_bus();
    gstocv->set_state(GST_STATE_PLAYING);
  
    const std::string modelPath = parser.get<std::string>("model_path");
    const std::string labelsPath = parser.get<std::string>("labels");
    const std::string detectorType = parser.get<std::string>("type");
    float confidenceThreshold = parser.get<float>("min_confidence");

    std::cout << "Current path is " << std::filesystem::current_path() << '\n'; // (1)

    std::unique_ptr<Detector> detector = createDetector(detectorType, labelsPath, modelPath); 
    if(!detector)
    {
        std::cerr << "Detector creation fail!" << std::endl;
        std::exit(1);
    }

    while(1) {
        gstocv->set_main_loop_event(false);
        cv::Mat frame = gstocv->get_frame().clone();
        if(!frame.empty())
        {
            std::vector<Detection> detections = detector->run_detection(frame);
            for(auto&& d : detections)
            {
                cv::rectangle(frame, d.bbox, cv::Scalar(255, 0, 0), 3);
            }
            cv::imshow("opencv feed", frame);  
            char key = cv::waitKey(1);
            if (key == 27 || key == 'q') // ESC
            {
                std::cout << "Exit requested" << std::endl;
                break;
            } 
        }
    }
    
    gstocv->set_state(GST_STATE_NULL);
    return 0;  
}
