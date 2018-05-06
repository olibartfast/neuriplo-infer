#ifndef TENSORFLOWOBJECTDETECTION_H
#define TENSORFLOWOBJECTDETECTION_H

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

using namespace cv;
using namespace dnn;
using namespace std;

class TensorFlowObjectDetection{
	const char** classNames_;
	Net net_;
    size_t inWidth_;
    size_t inHeight_;
    float inScaleFactor_;
    float meanVal_;
    float confidenceThreshold_;
public:
	void init(const char** classNames,
        String modelFile, 
        String configFile,     
        int frameWidth, 
        int frameHeight,        
        size_t inWidth = 300,
        size_t inHeight = 300,
        float inScaleFactor = 0.007843f,
        float meanVal = 127.5,
        float confidenceThreshold = 0.25);
	TensorFlowObjectDetection(){}


	void run_tf(Mat& frame);


};

#endif