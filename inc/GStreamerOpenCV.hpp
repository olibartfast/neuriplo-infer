#pragma once
#include "common.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <stdlib.h>
#include <glib.h>
#include <stdbool.h> 

class GStreamerOpenCV{
	GError *error_;
	GstElement *pipeline_;
	GstElement *sink_;
	GstBus *bus_;
	static cv::Mat frame_;

public:
	GStreamerOpenCV(GError *error);
	~GStreamerOpenCV();
	void init_gst_library(int argc, char *argv[]);
	void run_pipeline(std::string pipeline_cmd);
	void check_error();
	void get_sink();
	void set_bus();
	void set_state(GstState state);
	void set_main_loop_event(bool event);
	cv::Mat get_frame();
	void set_frame(cv::Mat &frame);
	static GstFlowReturn new_sample(GstAppSink *appsink, gpointer data);
	static GstFlowReturn new_preroll(GstAppSink *appsink, gpointer data);
};