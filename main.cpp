#include "GStreamerOpenCV.h"
#include "Detector.h"
    

static const string params = "{ help h   |   | print help message }"
      "{ arch     |  mobilenet | yolo, mobilenet or svm }"
      "{ link l   |   | capture video from ip camera}"
      "{ min_confidence | 0.5   | min confidence}";

int main (int argc, char *argv[])
{

    // Command line parser
    CommandLineParser parser(argc, argv, params);
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


    float confidenceThreshold = parser.get<float>("min_confidence");
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
  

    string architecture = parser.get<string>("arch");

    // OpenCV detection loop
    cvNamedWindow("opencv feed",1);
    Detector *detector = new Detector(architecture);
    while(1) {
        gstocv->set_main_loop_event(false);
        Mat frame = gstocv->get_frame();
        if(!frame.empty()){
            detector->run_detection(frame);
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
    delete detector;
    delete gstocv;
    return 0;  
}
