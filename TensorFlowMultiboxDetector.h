#include <setjmp.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <vector>

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


class TensorFlowMultiboxDetector{

public:

// Takes a file name, and loads a list of comma-separated box priors from it,
// one per line, and returns a vector of the values.
Status ReadLocationsFile(const string& file_name, std::vector<float>* result,
                         size_t* found_label_count);

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

// Analyzes the output of the MultiBox graph to retrieve the highest scores and
// their positions in the tensor, which correspond to individual box detections.
Status GetTopDetections(const std::vector<Tensor>& outputs, int how_many_labels,
Tensor* indices, Tensor* scores);

// Converts an encoded location to an actual box placement with the provided
// box priors.
void DecodeLocation(const float* encoded_location, const float* box_priors,
                    float* decoded_location);


inline float DecodeScore(float encoded_score) { return 1 / (1 + exp(-encoded_score)); }

void DrawBox(const int image_width, const int image_height, int left, int top,
             int right, int bottom, tensorflow::TTypes<uint8>::Flat* image);

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopDetections(const std::vector<Tensor>& outputs,
                          const string& labels_file_name,
                          const int num_boxes,
                          const int num_detections,
                          const string& image_file_name,
                          Tensor* original_tensor);


};



