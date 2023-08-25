#pragma once
#include "common.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <string>
#include <memory>

class GStreamerOpenCV {
    GError* error_ = nullptr;
    GstElement* pipeline_ = nullptr;
    GstElement* sink_ = nullptr;
    GstBus* bus_ = nullptr;
    inline static cv::Mat frame_;

    std::string getPipelineCommand(const std::string& link) const;

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

private:
    static GstFlowReturn newPreroll(GstAppSink* appsink, gpointer data);
    static GstFlowReturn newSample(GstAppSink* appsink, gpointer data);
    static gboolean myBusCallback(GstBus* bus, GstMessage* message, gpointer data);
};