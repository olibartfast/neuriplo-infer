#pragma once
#include <NvInfer.h>  // for TensorRT API
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // Implement logging behavior here, e.g., print the log message
        if (severity != Severity::kINFO)
        {
            std::cout << "TensorRT Logger: " << msg << std::endl;
        }
    }
};