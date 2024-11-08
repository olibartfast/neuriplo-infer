#pragma once

#include <opencv2/opencv.hpp>
struct Detection
{
	cv::Rect bbox;
	float score;
	int label;
};