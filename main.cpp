#include "HogSvmDetector.h"
#include "DnnDetector.h"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <stdlib.h>
#include <glib.h>
#include <stdbool.h> 

Mat frame;


 
GstFlowReturn
new_preroll(GstAppSink *appsink, gpointer data) {
    g_print ("Got preroll!\n");
    return GST_FLOW_OK;
}

GstFlowReturn
new_sample(GstAppSink *appsink, gpointer data) {
    static int framecount = 0;
    framecount++;
    static int width=0, height=0 ;
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    GstCaps *caps = gst_sample_get_caps(sample);
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    const GstStructure *info = gst_sample_get_info(sample);
    static GstStructure *s;
  s = gst_caps_get_structure(caps,0);
  gboolean res = gst_structure_get_int(s, "width", &width);
  res |= gst_structure_get_int(s, "height", &height);
  if(!res)
  {
    g_print("Could not get image width and height from filter caps");
    exit(-12);
  }


  // ---- Read frame and convert to opencv format ---------------

  GstMapInfo map;
  gst_buffer_map (buffer, &map, GST_MAP_READ);

  // convert gstreamer data to OpenCV Mat.

  Mat mYUV(height + height/2, width, CV_8UC1, (void*) map.data);
  Mat mRGB(height, width, CV_8UC3);
  cvtColor(mYUV, mRGB,  CV_YUV2RGBA_YV12, 3);
  mRGB.copyTo(frame);
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

static gboolean
my_bus_callback (GstBus *bus, GstMessage *message, gpointer data) {
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
      /* end-of-stream */
      break;
    default:
      /* unhandled message */
      break;
  }
  /* we want to be notified again the next time there is a message
   * on the bus, so returning TRUE (FALSE means we want to stop watching
   * for messages on the bus and our callback should not be called again)
   */
  return TRUE;
}



static const string keys = "{ help h   |   | print help message }"
                           "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
                           "{ link l   |   | capture video from ip camera}"
                           "{ model m     | | Path to a binary file of model contains trained weights. "
                                             "It could be a file with extensions .caffemodel (Caffe), "
                                             ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet) }"
                           "{ config c    | | Path to a text file of model contains network configuration. "
                                             "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet) }"
                           "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
                           "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
                           "{ mean        | | Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces. }"
                           "{ scale       |  1 | Preprocess input image by multiplying on a scale factor. }"
                           "{ width       | -1 | Preprocess input image by resizing to a specific width. }"
                           "{ height      | -1 | Preprocess input image by resizing to a specific height. }"
                           "{ rgb         |    | Indicate that model works with RGB input images instead BGR ones. }"
                           "{ thr         | .5 | Confidence threshold. }"
                           "{ backend     |  0 | Choose one of computation backends: "
                                                "0: default C++ backend, "
                                                "1: Halide language (http://halide-lang.org/), "
                                                "2: Intel's Deep Learning Inference Engine (https://software.seek.intel.com/deep-learning-deployment)}"
                           "{ target      |  0 | Choose one of target computation devices: "
                                                "0: CPU target (by default),"
                           			"1: OpenCL }";  	

int main (int argc, char *argv[])
{

  // Command line parser
  CommandLineParser parser(argc, argv, keys);
  parser.about("Detect people from rtsp ip camera stream");
  if (parser.has("help")){
    parser.printMessage();
    return 0;  
  }
  string link = parser.get<string>("link");
  if (!parser.check())
  {
        parser.printErrors();
        return 1;
  }
  if (link.empty()){
    cout << "Can not open video stream" << endl;
    return 2;
  }

  float confThreshold = parser.get<float>("thr");

    // Open file with classes names.
  std::vector<std::string> classes;
  if (parser.has("classes"))
  {
    std::string file = parser.get<String>("classes");
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
  }
  float scale = parser.get<float>("scale");
  Scalar mean = parser.get<Scalar>("mean");
  bool swapRB = parser.get<bool>("rgb");
  int inpWidth = parser.get<int>("width");
  int inpHeight = parser.get<int>("height");

  DnnDetector dnndetector;
  dnndetector.init(confThreshold, classes, scale, mean, swapRB, inpWidth, inpHeight);


  // Gstreamer
  string pipeline_cmd;
  pipeline_cmd = "rtspsrc location=" + link + " ! decodebin ! appsink name=autovideosink";
  printf("%s\n", pipeline_cmd.c_str());


  GError *error = NULL;

  gst_init (&argc, &argv);
  gchar *descr = g_strdup(pipeline_cmd.c_str());
  GstElement *pipeline = gst_parse_launch (descr, &error);

  if (error != NULL) {
    g_print ("could not construct pipeline: %s\n", error->message);
    g_error_free (error);
    exit (-1);
  }

  /* get sink */
  GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline), "autovideosink");

  gst_app_sink_set_emit_signals((GstAppSink*)sink, true);
  gst_app_sink_set_drop((GstAppSink*)sink, true);
  gst_app_sink_set_max_buffers((GstAppSink*)sink, 1);
  GstAppSinkCallbacks callbacks = { NULL, new_preroll, new_sample };
  gst_app_sink_set_callbacks (GST_APP_SINK(sink), &callbacks, NULL, NULL);

  GstBus *bus;
  guint bus_watch_id;
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, my_bus_callback, NULL);
  gst_object_unref (bus);

  gst_element_set_state (GST_ELEMENT (pipeline), GST_STATE_PLAYING);


  // OpenCV detection loop
  cvNamedWindow("opencv feed",1);
  HogSvmDetector hsdetector;
  while(1) {
        g_main_iteration(false);
        if(!frame.empty()){
	          frame = hsdetector.run_detection(frame);
            imshow("opencv feed", frame);  
            char key = waitKey(30);
            if (key == 27 || key == 'q') // ESC
            {
                cout << "Exit requested" << endl;
                break;
            }
      }
  }
 

  // Ending
  gst_element_set_state (GST_ELEMENT (pipeline), GST_STATE_NULL);
  gst_object_unref (GST_OBJECT (pipeline));

  return 0;  
}
