#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <iomanip>
#include <ctime>

using namespace cv;
using namespace dnn;
using namespace std;


class DnnDetector{
  float confThreshold;
  std::vector<std::string> classes;
  float scale;
  Scalar mean;
  bool swapRB;
  int inpWidth;
  int inpHeight;  
  Net net; 
public:
	DnnDetector(){}
    void init(float confThreshold_, 
    	std::vector<std::string> classes_, 
    	float scale_,
    	Scalar mean_, 
    	bool swapRB_, 
    	int inpWidth_, 
    	int inpHeight_);

    Mat run_dnn_detection(Mat frame);

    void postprocess(Mat& frame, const Mat& out, Net& net);

    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

    void callback(int pos, void* userdata);
};
