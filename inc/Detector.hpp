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
	std::vector<std::string> classNames_; 
  	std::string modelBinary_; 
	float confidenceThreshold_; 	  
  	size_t network_width_;
  	size_t network_height_;	
public:
	Detector(){}

	Detector(const Detector& d){
		classNames_ = d.classNames_;
  		modelBinary_ = d.modelBinary_; 
		confidenceThreshold_ = d.confidenceThreshold_;  
		network_width_ = d.network_width_;
  		network_height_ = d.network_height_;	
	}
	Detector(Detector&& d){
		classNames_ = d.classNames_;
  		modelBinary_ = d.modelBinary_; 
		confidenceThreshold_ = d.confidenceThreshold_;    
		network_width_ = d.network_width_;
  		network_height_ = d.network_height_;			
	}

	Detector(const std::vector<std::string>& classes, 
	const std::string& modelBinary,
	float confidenceThreshold = 0.5f, 
  	size_t network_width = -1,
  	size_t network_height = -1		
	) : classNames_{classes}, 
		modelBinary_{modelBinary},
		confidenceThreshold_{confidenceThreshold},
		network_width_ {network_width},
  		network_height_ {network_height}		 
	{

	}

    virtual std::vector<Detection> run_detection(const cv::Mat& frame) = 0;
};