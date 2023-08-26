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
	std::string backend_;
	bool use_gpu_;
	static std::shared_ptr<spdlog::logger> logger_; // Logger instance
public:
	Detector(const std::string& modelBinary,
	bool use_gpu = false, 
	float confidenceThreshold = 0.5f, 
  	size_t network_width = -1,
  	size_t network_height = -1		
	) : modelBinary_{modelBinary},
		use_gpu_{use_gpu},
		confidenceThreshold_{confidenceThreshold},
		network_width_ {network_width},
  		network_height_ {network_height}		 
	{
	}

	static void SetLogger(const std::shared_ptr<spdlog::logger>& logger) 
    {
    	logger_ = logger;
    }

    virtual std::vector<Detection> run_detection(const cv::Mat& frame) = 0;
};

std::shared_ptr<spdlog::logger> Detector::logger_;