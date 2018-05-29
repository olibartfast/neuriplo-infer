#ifndef YOLO_H
#define YOLO_H

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;

class Yolo{
	const char** coco_classes_;
	Net net_;
	size_t network_width_;
    size_t network_height_;
    float confidenceThreshold_;
public:
	Yolo(){}
	void init(const char** coco_classes, 
		string modelConfiguration, 
		string modelBinary, 
		float confidenceThreshold = 0.25,
		const size_t network_width = 416,
        const size_t network_height = 416
		);
	void run_yolo(Mat& frame);
};

#endif