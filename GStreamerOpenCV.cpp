#include "GStreamerOpenCV.h"


Mat GStreamerOpenCV::frame_;

GStreamerOpenCV::GStreamerOpenCV(GError *error){
    error_ = error;
}


GStreamerOpenCV::~GStreamerOpenCV(){
    gst_object_unref (GST_OBJECT (pipeline_));
    cout << "~GStreamerOpenCV()" << endl;
}


void GStreamerOpenCV::init_gst_library(int argc, char *argv[]){
    gst_init (&argc, &argv);

}

void GStreamerOpenCV::run_pipeline(string pipeline_cmd){		
	gchar *descr = g_strdup(pipeline_cmd.c_str());
	pipeline_ = gst_parse_launch (descr, &error_);
}

void GStreamerOpenCV::check_error(){
	if (error_ != NULL) {
        g_print ("could not construct pipeline: %s\n", error_->message);
        g_error_free (error_);
        exit (-1);
    }
}

Mat GStreamerOpenCV::get_frame() {
	return frame_; 
}

GstFlowReturn GStreamerOpenCV::new_preroll(GstAppSink *appsink, gpointer data) {
    g_print ("Got preroll!\n");
    return GST_FLOW_OK;
}

GstFlowReturn GStreamerOpenCV::new_sample(GstAppSink *appsink, gpointer data) {
    static int frameWidth=0, frameHeight=0 ;
    static int framecount = 0;
    framecount++;

    GstSample *sample = gst_app_sink_pull_sample(appsink);
    GstCaps *caps = gst_sample_get_caps(sample);
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    const GstStructure *info = gst_sample_get_info(sample);
    static GstStructure *s;
    s = gst_caps_get_structure(caps,0);
    gboolean res = gst_structure_get_int(s, "width", &frameWidth);
    res |= gst_structure_get_int(s, "height", &frameHeight);
    if(!res)
    {
        g_print("Could not get image width and height from filter caps");
        exit(-12);
    }
    
    // ---- Read frame and convert to opencv format ---------------
    GstMapInfo map;
    gst_buffer_map (buffer, &map, GST_MAP_READ);

    // convert gstreamer data to OpenCV Mat.
    Mat mYUV(frameHeight + frameHeight/2, frameWidth, CV_8UC1, (void*) map.data);
    Mat mRGB(frameHeight, frameWidth, CV_8UC3);
    cvtColor(mYUV, mRGB,  CV_YUV2RGBA_YV12, 3);
    mRGB.copyTo(GStreamerOpenCV::frame_);
    int frameSize = map.size;

    gst_buffer_unmap(buffer, &map);

    // ------------------------------------------------------------
    // print dot every 30 frames
    if (framecount%30 == 0) {
      g_print (".");
    }

    // show caps on first frame
    if (framecount == 1) {
      g_print ("%s\n", gst_caps_to_string(caps));
    }
    gst_sample_unref (sample);
    return GST_FLOW_OK;
}

static gboolean my_bus_callback (GstBus *bus, GstMessage *message, gpointer data) {
    g_print ("Got %s message\n", GST_MESSAGE_TYPE_NAME (message));
    switch (GST_MESSAGE_TYPE (message)) {
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;

            gst_message_parse_error (message, &err, &debug);
            g_print ("Error: %s\n", err->message);
            g_error_free (err);
            g_free (debug);    
            break;
        }
        case GST_MESSAGE_EOS:
            // end-of-stream 
            break;
        default:
            // unhandled message 
            break;
    }
    /* we want to be notified again the next time there is a message
    * on the bus, so returning TRUE (FALSE means we want to stop watching
    * for messages on the bus and our callback should not be called again)
    */
    return TRUE;
}

void GStreamerOpenCV::get_sink(){
	sink_ = gst_bin_get_by_name (GST_BIN (pipeline_), "autovideosink");
	gst_app_sink_set_emit_signals((GstAppSink*)sink_, true);
	gst_app_sink_set_drop((GstAppSink*)sink_, true);
	gst_app_sink_set_max_buffers((GstAppSink*)sink_, 1);
	static GstAppSinkCallbacks callbacks_ = { NULL, new_preroll, new_sample };
	gst_app_sink_set_callbacks (GST_APP_SINK(sink_), &callbacks_, NULL, NULL);
}


void GStreamerOpenCV::set_bus(){
	guint bus_watch_id;
	bus_ = gst_pipeline_get_bus (GST_PIPELINE (pipeline_));
	bus_watch_id = gst_bus_add_watch (bus_, my_bus_callback, NULL);
	gst_object_unref (bus_);
}


void GStreamerOpenCV::set_state(GstState state){
	gst_element_set_state (GST_ELEMENT (pipeline_), state);
}

void GStreamerOpenCV::set_main_loop_event(bool event){
	g_main_iteration(event);
}

