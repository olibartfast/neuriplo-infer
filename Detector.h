#include "HogSvmDetector.h"
#include "MobileNetSSD.h"
#include "Yolo.h"
#include "TensorFlowObjectDetection.h"

class Detector{
	 MobileNetSSD *mnssd_;
	 Yolo *yolo_;
	 HogSvmDetector *hsdetector_; 
	 TensorFlowObjectDetection *tfdetector_;   
	 string architecture_;
public:
	Detector(string architecture, float confidenceThreshold = 0.25, const int W = 1080, const int H = 720);
	~Detector();
    void run_detection(Mat& frame);
};