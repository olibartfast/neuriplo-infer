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
		backend_ = getInferenceBackend();
	}


	std::string getInferenceBackend()
	{
		const auto weights = modelBinary_;
		std::string backend;	
		size_t extension_pos = weights.rfind('.');
		if (extension_pos != std::string::npos)
		{
			std::string extension = weights.substr(extension_pos + 1);
			std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

			if (extension == "onnx")
			{
				// Use ONNX Runtime as the backend
				backend = "ONNX_RUNTIME";
			}
			else if (extension == "pt" || extension == "pth" || extension == "torchscript")
			{
				// Use LibTorch as the backend
				backend = "LIBTORCH";
			}
			else if (extension == "trt" || extension == "engine" || extension == "plan")
			{
				// Use TensorRT as the backend
				backend = "TENSORRT";
			}
			else
			{
				//logger->error("Invalid weight file extension. Supported extensions: .onnx, .pt, .pth, .trt");
				std::cout << "Invalid weight file extension. Supported extensions: .onnx, .pt, .pth, .trt" << std::endl;
				std::exit(1);
			}
		}
		else
		{
			//logger->error("Invalid weight file path");
			std::cout << "Invalid weight file path" << std::endl;
			std::exit(1);
		}
		return backend;
	}	
    virtual std::vector<Detection> run_detection(const cv::Mat& frame) = 0;
};