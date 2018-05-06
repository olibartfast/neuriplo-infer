#ifndef MOBILENETSSD_H
#define MOBILENETSSD_H
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
  ~MobileNetSSD(){
    cout << "~MobileNetSSD()" << endl;
  }
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
#endif