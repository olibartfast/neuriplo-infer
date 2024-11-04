#pragma once
#include "common.hpp"
#include <variant>


// First, define the variant type (could be in the header file)
using TensorElement = std::variant<float, int32_t, int64_t>;



struct Detection
{
	cv::Rect bbox;
	float score;
	int label;
};

class Detector{
protected:	
	float confidenceThreshold_; 
	float nms_threshold_ = 0.4f;	  
  	size_t network_width_;
  	size_t network_height_;	
	std::string backend_;
    int channels_{ -1 };
public:
	Detector(
	float confidenceThreshold = 0.5f, 
  	size_t network_width = -1,
  	size_t network_height = -1		
	) :	confidenceThreshold_{confidenceThreshold},
		network_width_ {network_width},
  		network_height_ {network_height}		 
	{
	}

    inline float getConfidenceThreshold(){ return confidenceThreshold_; }
    inline float getNetworkWidth() { return network_width_; }
    inline float getNetworkHeight() { return network_height_; } 

	virtual std::vector<Detection> postprocess(const std::vector<std::vector<TensorElement>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size) = 0;
    virtual cv::Mat preprocess_image(const cv::Mat& image) = 0; 


};
