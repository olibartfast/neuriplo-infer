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

// MobileNet Single-Shot Detector 
// (https://arxiv.org/abs/1512.02325)
// to detect objects on image, caffemodel model's file is avaliable here:
// https://github.com/chuanqi305/MobileNet-SSD


class MobileNetSSD{
  const char** classNames_;
  size_t inWidth_;
  size_t inHeight_;
  float inScaleFactor_;
  float meanVal_;
  float confidenceThreshold_;
  String modelConfiguration_;
  String modelBinary_;  
  Net net_; 
  Rect crop_;
  VideoWriter outputVideo_;

public:
	MobileNetSSD(){}
  ~MobileNetSSD(){
    cout << "~MobileNetSSD()" << endl;
  }
    void init(const char** classNames,
        String modelConfiguration, 
        String modelBinary,     
        int frameWidth, 
        int frameHeight,        
        size_t inWidth = 300,
        size_t inHeight = 300,
        float inScaleFactor = 0.007843f,
        float meanVal = 127.5,
        float confidenceThreshold = 0.25);
    void run_ssd(Mat& frame);

};
#endif