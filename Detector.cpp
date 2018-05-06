#include "Detector.h"
Detector::Detector(string architecture, float confidenceThreshold, const int W, const int H){
	architecture_ = architecture;
	yolo_ = NULL;
	hsdetector_ = NULL;   
	mnssd_ = NULL; 
    if(architecture == "mobilenet"){
        // Open file with classes names.
        static const char* classNames[] = {"background",
                                    "aeroplane", "bicycle", "bird", "boat",
                                    "bottle", "bus", "car", "cat", "chair",
                                    "cow", "diningtable", "dog", "horse",
                                    "motorbike", "person", "pottedplant",
                                    "sheep", "sofa", "train", "tvmonitor"};    

        String modelConfiguration = "models/MobileNetSSD_deploy.prototxt";
        String modelBinary = "models/MobileNetSSD_deploy.caffemodel";
        mnssd_ = new MobileNetSSD();
        mnssd_->init(classNames, modelConfiguration, modelBinary, W, H);

    }
    else if (architecture.find("yolo") != string::npos){
    	yolo_ = new Yolo();

        // get the model and cfg files here: https://pjreddie.com/darknet/yolo/
        String modelConfiguration;
        String modelBinary;
        if(architecture == "yolov2-tiny"){
        	modelConfiguration = "models/yolov2-tiny.cfg";
        	modelBinary = "models/yolov2-tiny.weights";
        }
        if(architecture == "yolov2"){
        	modelConfiguration = "models/yolov2.cfg";
        	modelBinary = "models/yolov2.weights";
        }
        if(architecture == "yolov3"){
        	modelConfiguration = "models/yolov3.cfg";
        	modelBinary = "models/yolov3.weights";
        }
        yolo_->init(modelConfiguration, modelBinary);


    }
		
        
    else if(architecture == "svm")
    	hsdetector_ = new HogSvmDetector();
}

Detector::~Detector(){
	if(mnssd_ != NULL)
		delete mnssd_;
	if(yolo_ != NULL)
		delete yolo_;
	if(hsdetector_ != NULL)
		delete hsdetector_;
	cout << "~Detector()" << endl;

}

void Detector::run_detection(Mat& frame){
	if(architecture_ == "mobilenet")
        mnssd_->run_ssd(frame);
    else if (architecture_.find("yolo") != string::npos)
    	yolo_->run_yolo(frame);
    else if(architecture_ == "svm")	
        hsdetector_->run_detection(frame);
}