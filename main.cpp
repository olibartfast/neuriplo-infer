#include "GStreamerOpenCV.hpp"
#include "Detector.hpp"
#include "HogSvmDetector.hpp"
#include "MobileNetSSD.hpp"
#include "Yolo.hpp"

static const std::string params = "{ help h   |   | print help message }"
      "{ type     |  yolov4 | mobilenet, svm, yolov4-tiny, yolov4}"
      "{ link l   |   | capture video from ip camera}"
      "{ labels lb   |  ../labels | path to class labels file}"
      "{ model_path mp   |  ../models | path to models}"
      "{ min_confidence | 0.25   | min confidence}";

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
    if(!std::filesystem::exists(modelConfiguration) || !std::filesystem::exists(modelBinary)){
        std::cerr << "Wrong path to model " << std::endl;
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
    else if(detectorType == "yolov4" || detectorType == "yolov4-tiny")
    {
        std::string modelConfiguration, modelBinary;
        classes = readLabelNames(labelsPath + "/" + "coco.names"); 
        if(detectorType == "yolov4-tiny")
            std::tie(modelConfiguration, modelBinary) = modelSetup(modelPath, "yolov4-tiny.cfg",  "yolov4-tiny.weights");
        else if(detectorType == "yolov4")
        	std::tie(modelConfiguration, modelBinary) = modelSetup(modelPath, "yolov4.cfg",  "yolov4.weights");    
        return std::make_unique<Yolo>(classes, modelConfiguration, modelBinary);
    }    
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



    GError *error = NULL;
    GStreamerOpenCV *gstocv = new GStreamerOpenCV(error);
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

    while(1) {
        gstocv->set_main_loop_event(false);
        cv::Mat frame = gstocv->get_frame().clone();
        if(!frame.empty())
        {
            detector->run_detection(frame);
            imshow("opencv feed", frame);  
            char key = cv::waitKey(1);
            if (key == 27 || key == 'q') // ESC
            {
                std::cout << "Exit requested" << std::endl;
                break;
            } 
        }
    }
    
    gstocv->set_state(GST_STATE_NULL);
    delete gstocv;
    return 0;  
}
