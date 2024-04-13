#pragma once
#include "common.hpp"

class InferenceInterface{
    	
    public:
        InferenceInterface(const std::string& weights, const std::string& modelConfiguration,
         bool use_gpu = false)
        {

        }



        static void SetLogger(const std::shared_ptr<spdlog::logger>& logger) 
        {
            logger_ = logger;
        }

        
        virtual std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) = 0;

    protected:
        std::vector<float> blob2vec(const cv::Mat& input_blob);
        size_t get_blob_size(const cv::Mat& input_blob);
        static std::shared_ptr<spdlog::logger> logger_; 



};