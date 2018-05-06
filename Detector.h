#include "HogSvmDetector.h"
#include "MobileNetSSD.h"
#include "Yolo.h"

class Detector{
	 MobileNetSSD *mnssd_;
	 Yolo *yolo_;
	 HogSvmDetector *hsdetector_;    
	 string architecture_;
	public:
		Detector(string architecture, float confidenceThreshold = 0.20, const int W = 1080, const int H = 720){
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
		        size_t inWidth = 300;
		        size_t inHeight = 300;
		        float inScaleFactor = 0.007843f;
		        float meanVal = 127.5;
		        String modelConfiguration = "models/MobileNetSSD_deploy.prototxt";
		        String modelBinary = "models/MobileNetSSD_deploy.caffemodel";
		        mnssd_ = new MobileNetSSD();
		        mnssd_->init(classNames, 
		              inWidth, inHeight, 
		              inScaleFactor, meanVal, 
		              W, H, 
		              confidenceThreshold, 
		              modelConfiguration, modelBinary);

		    }
		    else if(architecture == "yolo")
        		yolo_ = new Yolo();
            else if(architecture == "svm")
            	hsdetector_ = new HogSvmDetector();
		}

		~Detector(){
			if(mnssd_ != NULL)
				delete mnssd_;
			if(yolo_ != NULL)
				delete yolo_;
			if(hsdetector_ != NULL)
				delete hsdetector_;
			cout << "~Detector()" << endl;

		}

		void run_detection(Mat& frame){
			if(architecture_ == "mobilenet")
                frame = mnssd_->run_ssd(frame);
            else if(architecture_ == "yolo")
            	((void)0);
            else if(architecture_ == "svm")	
	            frame = hsdetector_->run_detection(frame);
		}
};