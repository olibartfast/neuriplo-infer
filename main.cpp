#include "HogSvmDetector.h"
#include "MobileNetSSD.h"
#include "GStreamerOpenCV.h"
    

// ip camera frame size
const int W = 1080;
const int H = 720;


static const string keys = "{ help h   |   | print help message }"
      "{ link l   |   | capture video from ip camera}"
      "{ proto          | models/MobileNetSSD_deploy.prototxt | model configuration }"
      "{ model          | models/MobileNetSSD_deploy.caffemodel | model weights }"
      "{ video          |       | video for detection }"
      "{ out            |       | path to output video file}"
      "{ min_confidence | 0.5   | min confidence      }";

int main (int argc, char *argv[])
{

    // Command line parser
    CommandLineParser parser(argc, argv, keys);
    parser.about("Detect people from rtsp ip camera stream");
    if (parser.has("help")){
      parser.printMessage();
      return 0;  
    }
    string link = parser.get<string>("link");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    if (link.empty()){
        cout << "Can not open video stream" << endl;
        return 2;
    }


  
    GError *error = NULL;
    GStreamerOpenCV *gstocv = new GStreamerOpenCV(error);
    gstocv->init_gst_library(argc, argv);
    string pipeline_cmd = "rtspsrc location=" + link + " ! decodebin ! appsink name=autovideosink";
    cout << pipeline_cmd.c_str() << endl;
    gstocv->run_pipeline(pipeline_cmd);
    gstocv->check_error();
    gstocv->get_sink();
    gstocv->set_bus();
    gstocv->set_state(GST_STATE_PLAYING);
  

    // OpenCV detection loop
    cvNamedWindow("opencv feed",1);

    #ifdef DNN 
    // Open file with classes names.
    const char* classNames[] = {"background",
                                "aeroplane", "bicycle", "bird", "boat",
                                "bottle", "bus", "car", "cat", "chair",
                                "cow", "diningtable", "dog", "horse",
                                "motorbike", "person", "pottedplant",
                                "sheep", "sofa", "train", "tvmonitor"};

    size_t inWidth = 300;
    size_t inHeight = 300;
    float inScaleFactor = 0.007843f;
    float meanVal = 127.5;
    float confidenceThreshold = parser.get<float>("min_confidence");
    String modelConfiguration = parser.get<string>("proto");
    String modelBinary = parser.get<string>("model");
    MobileNetSSD mnssd;

    mnssd.init(classNames, 
      inWidth, inHeight, 
      inScaleFactor, meanVal, 
      W, H, 
      confidenceThreshold, 
      modelConfiguration, modelBinary);
    #else
    HogSvmDetector hsdetector;
    #endif  
    while(1) {
          gstocv->set_main_loop_event(false);
          Mat frame = gstocv->get_frame();
          if(!frame.empty()){
              #ifdef DNN 
              frame = mnssd.run_ssd(frame);
              #else
	            frame = hsdetector.run_detection(frame);
              #endif
              imshow("opencv feed", frame);  
              char key = waitKey(50);
              if (key == 27 || key == 'q') // ESC
              {
                  cout << "Exit requested" << endl;
                  break;
              }
          }
    }
 

    // Ending
    gstocv->set_state(GST_STATE_NULL);
    delete gstocv;
    return 0;  
}
