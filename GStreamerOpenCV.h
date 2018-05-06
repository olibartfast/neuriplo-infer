#ifndef GSTREAMEROPENCV_H
#define GSTREAMEROPENCV_H

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <stdlib.h>
#include <glib.h>
#include <stdbool.h> 
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>


using namespace std;
using namespace cv;

class GStreamerOpenCV{
	GError *error_;
	GstElement *pipeline_;
	GstElement *sink_;
	GstBus *bus_;
	static Mat frame_;

public:
	GStreamerOpenCV(GError *error);
	~GStreamerOpenCV();
	void init_gst_library(int argc, char *argv[]);
	void run_pipeline(string pipeline_cmd);
	void check_error();
	void get_sink();
	void set_bus();
	void set_state(GstState state);
	void set_main_loop_event(bool event);
	Mat get_frame();
	void set_frame(Mat &frame);
	static GstFlowReturn new_sample(GstAppSink *appsink, gpointer data);
	static GstFlowReturn new_preroll(GstAppSink *appsink, gpointer data);
};
#endif