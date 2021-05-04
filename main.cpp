#include "GStreamerOpenCV.hpp"
#include "Detector.hpp"
#include "HogSvmDetector.hpp"
#include "MobileNetSSD.hpp"
#include "Yolo.hpp"

//TensorFlowObjectDetection *tfodetector_;  
//TensorFlowMultiboxDetector *tfmbdetector_; 

static const std::string params = "{ help h   |   | print help message }"
      "{ type     |  yolov2 | mobilenet, svm, yolov2-tiny, yolov2}"
      "{ link l   |   | capture video from ip camera}"
      "{ min_confidence | 0.5   | min confidence}";

std::vector<std::string> readLabelNames(const std::string& fileName)
{
    std::vector<std::string> classes;
    std::ifstream ifs(fileName.c_str());
    std::string line;
    while (getline(ifs, line))
    classes.push_back(line);
    return classes;   
}

std::unique_ptr<Detector> createDetector(const std::string& detectorType){
    std::vector<std::string> classes; 
    std::string modelConfiguration; 
    std::string modelBinary;  
    if(detectorType == "svm"){
        return std::make_unique<HogSvmDetector>();
    }
    else if(detectorType == "mobilenet")
    {
        classes = readLabelNames("voc.names"); 
        modelConfiguration = "models/MobileNetSSD_deploy.prototxt"; 
        modelBinary  = "models/MobileNetSSD_deploy.caffemodel";  
        return std::make_unique<MobileNetSSD>(classes, modelConfiguration, modelBinary);
    }
    else if(detectorType == "yolov2" || detectorType == "yolov2-tiny")
    {
        classes = readLabelNames("coco.names"); 
        if(detectorType == "yolov2-tiny"){
        	modelConfiguration = "models/yolov2-tiny.cfg";
        	modelBinary = "models/yolov2-tiny.weights";
        }
        else if(detectorType == "yolov2"){
        	modelConfiguration = "models/yolov2.cfg";
        	modelBinary = "models/yolov2.weights";
        }        
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
  

    const std::string detectorType = parser.get<std::string>("type");
    float confidenceThreshold = parser.get<float>("min_confidence");

    std::cout << "Current path is " << std::filesystem::current_path() << '\n'; // (1)

    std::unique_ptr<Detector> detector = createDetector(detectorType); 

    while(1) {
         gstocv->set_main_loop_event(false);
         cv::Mat frame = gstocv->get_frame();
        if(!frame.empty()){
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
