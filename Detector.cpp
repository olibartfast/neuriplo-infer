#include "Detector.h"
Detector::Detector(String architecture, float confidenceThreshold, const int W, const int H){
	architecture_ = architecture;
	yolo_ = NULL;
	hsdetector_ = NULL;   
	mnssd_ = NULL; 
    //tfdetector_ = NULL;
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
        mnssd_->init(classNames, modelConfiguration, modelBinary, W, H, confidenceThreshold );

    }
    else if ( architecture == "yolov2-tiny" ||
    	architecture == "yolov2"){
    	yolo_ = new Yolo();
    	cout << "Yolo found" << endl;
        // get the model and cfg files here: https://pjreddie.com/darknet/yolo/
        String modelConfiguration;
        String modelBinary;

        static const char *coco_classes[] = {"person","bicycle","car","motorcycle","airplane","bus","train",
        "truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird",
        "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
        "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
        "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
        "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
        "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv",
        "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

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
        yolo_->init(coco_classes, modelConfiguration, modelBinary, confidenceThreshold);


    }
		
        
    else if(architecture == "svm")
    	hsdetector_ = new HogSvmDetector();
    /*else if(architecture == "tensorflow"){
        tfdetector_ = new TensorFlowObjectDetection();
        static const char* classNames[] = {"background",
                            "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"}; 
        String modelFile = "models/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb";
        String configFile = "models/ssd_mobilenet_v1_coco_11_06_2017/ssd_mobilenet_v1_coco.pbtxt";
        tfdetector_->init(classNames, modelFile, configFile, W, H);
    }*/
    else if(architecture == "tf-multibox-detector"){
        tfmbdetector_ = new TensorFlowMultiboxDetector();
        string graph = "models/multibox_detector/multibox_model.pb";
        string box_priors = "models/multibox_detector/multibox_location_priors.txt";
        tfmbdetector_->init(graph, box_priors);
    }
}

Detector::~Detector(){
	if(mnssd_ != NULL)
		delete mnssd_;
	if(yolo_ != NULL)
		delete yolo_;
	if(hsdetector_ != NULL)
		delete hsdetector_;
    //if(tfdetector_  != NULL)
    //    delete tfdetector_;
    if(tfmbdetector_ != NULL)
       delete  tfmbdetector_;
	cout << "~Detector()" << endl;

}

void Detector::run_detection(Mat& frame){
	if(architecture_ == "mobilenet")
        mnssd_->run_ssd(frame);
    else if( architecture_ == "yolov2-tiny" ||
    	architecture_ == "yolov2" ||
    	architecture_ == "yolov3")
    	yolo_->run_yolo(frame);
    else if(architecture_ == "svm")	
        hsdetector_->run_detection(frame);
    //else if(architecture_ == "tensorflow")
    //    tfdetector_ ->run_tf(frame);
    else if(architecture_ == "tf-multibox-detector")
        tfmbdetector_->run_multibox_detector(frame);

}