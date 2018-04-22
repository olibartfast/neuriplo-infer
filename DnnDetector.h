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
public:
    DnnDetector(){}

    Mat run_dnn_detection(Mat frame);

    void postprocess(Mat& frame, const Mat& out, Net& net);

    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

    void callback(int pos, void* userdata);
};
