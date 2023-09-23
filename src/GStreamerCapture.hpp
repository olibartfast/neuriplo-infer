#pragma once
#include "VideoCaptureInterface.hpp"
#include "GStreamerOpenCV.hpp"
#include <condition_variable>
#include <mutex>

class GStreamerCapture : public VideoCaptureInterface {
private:
    GStreamerOpenCV gstocv;
    bool initialized = false; // Track initialization status
    std::mutex frameMutex_; // Mutex to protect frame access
    std::condition_variable frameAvailable_; // Condition variable to signal new frames

public:
    bool initialize(const std::string& source) {
        gstocv.initGstLibrary(0, nullptr);
        gstocv.runPipeline(source);
        gstocv.checkError();
        gstocv.getSink();
        gstocv.setBus();
        gstocv.setState(GST_STATE_PLAYING);
        initialized = true;
        return true;
    }

    bool readFrame(cv::Mat& frame) override {
        if (!initialized ||   GStreamerOpenCV::isEndOfStream()) {
            // Handle attempts to read frames without proper initialization
            return false;
        }   
        gstocv.setMainLoopEvent(false);

        {
            std::unique_lock<std::mutex> lock(GStreamerOpenCV::frameMutex_);
            GStreamerOpenCV::frameAvailable_.wait(lock, [this] { return GStreamerOpenCV::isFrameReady_; });
            frame = gstocv.getFrame().clone();
        }   
        return !frame.empty();
    }

    void release() override {
        // Release GStreamer resources
        gstocv.setState(GST_STATE_NULL);

        // Reset the initialization status
        initialized = false;
    }
};