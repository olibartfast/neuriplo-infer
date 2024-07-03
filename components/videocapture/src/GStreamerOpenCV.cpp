#include "GStreamerOpenCV.hpp"

std::mutex GStreamerOpenCV::frameMutex_;
std::condition_variable  GStreamerOpenCV::frameAvailable_; 
cv::Mat GStreamerOpenCV::frame_;
bool GStreamerOpenCV::end_of_stream_ = false;
bool GStreamerOpenCV::isFrameReady_ = false;


GStreamerOpenCV::GStreamerOpenCV() {
    error_ = nullptr;
}

void GStreamerOpenCV::setEndOfStream(bool value) {
    end_of_stream_ = value;
}

bool GStreamerOpenCV::isEndOfStream() {
    return end_of_stream_;
}

GStreamerOpenCV::~GStreamerOpenCV() {
    if (pipeline_) {
        gst_object_unref(GST_OBJECT(pipeline_));
        pipeline_ = nullptr;
    }
}

void GStreamerOpenCV::initGstLibrary(int argc, char* argv[]) {
    gst_init(&argc, &argv);
}

void GStreamerOpenCV::runPipeline(const std::string& link) {
    const std::string pipelineCmd = getPipelineCommand(link);
    gchar* descr = g_strdup(pipelineCmd.c_str());
    pipeline_ = gst_parse_launch(descr, &error_);
    g_free(descr);
}

void GStreamerOpenCV::checkError() {
    if (error_ != nullptr) {
        g_print("Could not construct pipeline: %s\n", error_->message);
        g_error_free(error_);
        error_ = nullptr;
        exit(-1);
    }
}

cv::Mat GStreamerOpenCV::getFrame() const {
    return frame_;
}

void GStreamerOpenCV::setFrame(const cv::Mat& frame) {
    frame.copyTo(frame_);
}

std::string GStreamerOpenCV::getPipelineCommand(const std::string& link) const {
    if (link.find("rtsp") != std::string::npos)
        return "rtspsrc location=" + link + " ! decodebin ! appsink name=autovideosink";
    else
        return "filesrc location=" + link + " ! decodebin ! appsink name=autovideosink";
}

GstFlowReturn GStreamerOpenCV::newPreroll(GstAppSink* appsink, gpointer data) {
    g_print("Got preroll!\n");
    return GST_FLOW_OK;
}

GstFlowReturn GStreamerOpenCV::newSample(GstAppSink* appsink, gpointer data) {
    isFrameReady_ = false;
    static int frameWidth = 0, frameHeight = 0;
    static int framecount = 0;
    framecount++;

    GstSample* sample = gst_app_sink_pull_sample(appsink);
    GstCaps* caps = gst_sample_get_caps(sample);
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    const GstStructure* info = gst_sample_get_info(sample);
    const GstStructure* s = gst_caps_get_structure(caps, 0);

    gboolean res = gst_structure_get_int(s, "width", &frameWidth);
    res |= gst_structure_get_int(s, "height", &frameHeight);
    if (!res) {
        g_print("Could not get image width and height from filter caps\n");
        return GST_FLOW_OK;
    }

    // Read frame and convert to OpenCV format
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    // Convert GStreamer data to OpenCV Mat
    cv::Mat mYUV(frameHeight + frameHeight / 2, frameWidth, CV_8UC1, map.data);
    cv::Mat mRGB(frameHeight, frameWidth, CV_8UC3);
    cvtColor(mYUV, mRGB, cv::COLOR_YUV2RGBA_YV12, 3);
  
    // Signal that a new frame is available
    {
        std::lock_guard<std::mutex> lock(frameMutex_);
        mRGB.copyTo(GStreamerOpenCV::frame_);
        isFrameReady_ = true;
        frameAvailable_.notify_one();
    }

    int frameSize = map.size;
    gst_buffer_unmap(buffer, &map);

    // Show caps on the first frame
    if (framecount == 1) {
        g_print("%s\n", gst_caps_to_string(caps));
    }

    gst_sample_unref(sample);




    return GST_FLOW_OK;
}

gboolean GStreamerOpenCV::myBusCallback(GstBus* bus, GstMessage* message, gpointer data) {
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR: {
            GError* err;
            gchar* debug;
            gst_message_parse_error(message, &err, &debug);
            g_print("Error: %s\n", err->message);
            g_error_free(err);
            g_free(debug);
            break;
        }
        case GST_MESSAGE_EOS:{
			g_message ("End of stream");
            setEndOfStream(true); 
            break;
        }

        default:
            // Unhandled message
            break;
    }

    // Return TRUE to be notified again the next time there is a message on the bus
    return TRUE;
}

void GStreamerOpenCV::getSink() {
    sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "autovideosink");
    gst_app_sink_set_emit_signals(GST_APP_SINK(sink_), true);
    gst_app_sink_set_drop(GST_APP_SINK(sink_), true);
    gst_app_sink_set_max_buffers(GST_APP_SINK(sink_), 1);
    GstAppSinkCallbacks callbacks = { nullptr, newPreroll, newSample };
    gst_app_sink_set_callbacks(GST_APP_SINK(sink_), &callbacks, nullptr, nullptr);
}

void GStreamerOpenCV::setBus() {
    bus_ = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
    gst_bus_add_watch(bus_, myBusCallback, nullptr);
    gst_object_unref(bus_);
}

void GStreamerOpenCV::setState(GstState state) {
    gst_element_set_state(GST_ELEMENT(pipeline_), state);
}

void GStreamerOpenCV::setMainLoopEvent(bool event) {
    g_main_iteration(event);
}