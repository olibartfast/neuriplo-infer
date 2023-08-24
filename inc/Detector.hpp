#pragma once
#include "common.hpp"

struct Detection
{
	cv::Rect bbox;
	float score;
	int label;
};

class Detector{
protected:	
  	std::string modelBinary_; 
	float confidenceThreshold_; 	  
  	size_t network_width_;
  	size_t network_height_;	
public:

	Detector(const std::string& modelBinary,
	float confidenceThreshold = 0.5f, 
  	size_t network_width = -1,
  	size_t network_height = -1		
	) : modelBinary_{modelBinary},
		confidenceThreshold_{confidenceThreshold},
		network_width_ {network_width},
  		network_height_ {network_height}		 
	{

	}
    virtual std::vector<Detection> run_detection(const cv::Mat& frame) = 0;
};