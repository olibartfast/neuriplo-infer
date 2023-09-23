#pragma once
#include "common.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <string>
#include <memory>
#include <condition_variable>
#include <mutex>

class GStreamerOpenCV {


public:
    GStreamerOpenCV();
    ~GStreamerOpenCV();
    void initGstLibrary(int argc, char* argv[]);
    void runPipeline(const std::string& link);
    void checkError();
    void getSink();
    void setBus();
    void setState(GstState state);
    void setMainLoopEvent(bool event);
    cv::Mat getFrame() const;
    void setFrame(const cv::Mat& frame);

    static void setEndOfStream(bool value);
    static bool isEndOfStream();


    static std::mutex frameMutex_; // Mutex to protect frame access
    static std::condition_variable frameAvailable_; // Condition variable to signal new frames
    static bool isFrameReady_;

private:
    static GstFlowReturn newPreroll(GstAppSink* appsink, gpointer data);
    static GstFlowReturn newSample(GstAppSink* appsink, gpointer data);
    static gboolean myBusCallback(GstBus* bus, GstMessage* message, gpointer data);

    GError* error_ = nullptr;
    GstElement* pipeline_ = nullptr;
    GstElement* sink_ = nullptr;
    GstBus* bus_ = nullptr;
    static cv::Mat frame_;
    static bool end_of_stream_;



    std::string getPipelineCommand(const std::string& link) const;

};