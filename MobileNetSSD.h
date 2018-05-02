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

// MobileNet Single-Shot Detector 
// (https://arxiv.org/abs/1512.02325)
// to detect objects on image, caffemodel model's file is avaliable here:
// https://github.com/chuanqi305/MobileNet-SSD


class MobileNetSSD{
  const char** classNames;
  size_t inWidth;
  size_t inHeight;
  float WHRatio;
  float inScaleFactor;
  float meanVal;
  int frameWidth; 
  int frameHeight;   
  float confidenceThreshold;
  String modelConfiguration;
  String modelBinary;  
  Net net; 
  Size inVideoSize;
  Size cropSize;
  Rect crop;
  VideoWriter outputVideo;

public:
	MobileNetSSD(){}
    void init(const char** classNames_,
    	size_t inWidth_, 
    	size_t inHeight_, 
    	float inScaleFactor_,
    	float meanVal_, 
 		int frameWidth_, 
 		int frameHeight_,   	
 		float confidenceThreshold_,
    	String modelConfiguration_, 
    	String modelBinary_);
    Mat run_ssd(Mat frame);

};
