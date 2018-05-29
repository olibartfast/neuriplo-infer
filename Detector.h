#include "HogSvmDetector.h"
#include "MobileNetSSD.h"
#include "Yolo.h"
#include "TensorFlowMultiboxDetector.h"
//#include "TensorFlowObjectDetection.h"

class Detector{
	 MobileNetSSD *mnssd_;
	 Yolo *yolo_;
	 HogSvmDetector *hsdetector_; 
	 //TensorFlowObjectDetection *tfdetector_;  
	 TensorFlowMultiboxDetector *tfmbdetector_; 
	 string architecture_;
public:
	Detector(String architecture, float confidenceThreshold, const int W = 1080, const int H = 720);
	~Detector();
    void run_detection(Mat& frame);
};