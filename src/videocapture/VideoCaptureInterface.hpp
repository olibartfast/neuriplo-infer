#pragma once
#include <opencv2/core/core.hpp>

class VideoCaptureInterface {
public:
    virtual ~VideoCaptureInterface() {}

    // Initialize the video capture from a source (e.g., file, camera, URL).
    virtual bool initialize(const std::string& source) = 0;

    // Read a frame from the video source.
    virtual bool readFrame(cv::Mat& frame) = 0;

    // Release any resources associated with the video capture.
    virtual void release() = 0;
};