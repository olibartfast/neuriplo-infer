

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


class HogSvmDetector
{
    enum Mode { Default, Daimler } m;
    HOGDescriptor hog, hog_d;
public:
    HogSvmDetector() : m(Daimler), hog(), hog_d(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9)
    {
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        hog_d.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    }    
    string modeName() const;
    vector<Rect> detect(InputArray img);
    void adjustRect(Rect & r) const;
    Mat run_detection(Mat frame);

};

