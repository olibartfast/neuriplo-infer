#ifndef TENSORFLOWOBJECTDETECTION_H
#define TENSORFLOWOBJECTDETECTION_H

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>


#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::uint8;

using namespace cv;
using namespace dnn;
using namespace std;

class TensorFlowObjectDetection{
    string graph_;
	  string labels_;
    string image_;
    int32 input_width_;
    int32 input_height_;
    int32 input_mean_;
    int32 input_std_;

    string input_layer_;
    std::vector<string> output_layer_;
    string root_dir_;
 
    std::unique_ptr<tensorflow::Session> session_;
    std::vector<Tensor> image_tensors_;

    string graph_path_;
    string labels_path_;

    int frameWidth_;
    int frameHeight_;      
    float confidenceThreshold_;

    const char** coco_classes_;





public:
TensorFlowObjectDetection(){}
void init(
    const char** coco_classes,
    string graph,
    string labels,
    int frameWidth, 
    int frameHeight,      
    float confidenceThreshold = 0.25,
    int32 input_width = 300,
    int32 input_height = 300,
    int32 input_mean = 127.5,
    int32 input_std = 255); 



// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors);

Status SaveImage(const Tensor& tensor, const string& file_path);


// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session);



void run_tf_object_detection(Mat& frame);


};


#endif