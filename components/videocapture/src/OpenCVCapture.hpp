#pragma once
#include "VideoCaptureInterface.hpp"
#include <opencv2/highgui/highgui.hpp>


class OpenCVCapture : public VideoCaptureInterface {
private:
    cv::VideoCapture capture;
    bool initialized = false; // Track initialization status

public:
    bool initialize(const std::string& source) override {
        // Initialize OpenCV video capture
        if (!capture.open(source)) {
            // Handle initialization errors
            initialized = false;
            return false;
        }

        initialized = true;
        return true;
    }

    bool readFrame(cv::Mat& frame) override {
        if (!initialized) {
            // Handle attempts to read frames without proper initialization
            return false;
        }

        return capture.read(frame);
    }

    void release() override {
        // Release OpenCV video capture resources
        capture.release();

        // Reset the initialization status
        initialized = false;
    }
};