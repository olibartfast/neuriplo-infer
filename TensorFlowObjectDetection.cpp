#include "TensorFlowObjectDetection.h"




void TensorFlowObjectDetection::init(
    const char** coco_classes,
    string graph,
    string labels,
    int frameWidth, 
    int frameHeight,      
    float confidenceThreshold,    
    int32 input_width,
    int32 input_height,
    int32 input_mean,
    int32 input_std){

    graph_ = graph;
    labels_ = labels;
    frameWidth_ = frameWidth;
    frameHeight_ = frameHeight;      
    confidenceThreshold_ = confidenceThreshold;    
    input_width_ = input_width;
    input_height_ = input_height;
    input_mean_ = input_mean;
    input_std_ = input_std;
    coco_classes_ = coco_classes;

    
    input_layer_ = "image_tensor:0";
    output_layer_ ={ "detection_boxes:0", 
        "detection_scores:0", 
        "detection_classes:0", 
        "num_detections:0" };
    root_dir_ = "";

    // First we load and initialize the model.
    graph_path_ = tensorflow::io::JoinPath(root_dir_, graph_);
    labels_path_ = tensorflow::io::JoinPath(root_dir_, labels_);
    Status load_graph_status = LoadGraph(graph_path_, &session_);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return;
    }

 

}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status TensorFlowObjectDetection::ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors){
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };


  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    image_reader = DecodeGif(root.WithOpName("gif_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }

  // Also return identity so that we can know the original dimensions and
  // optionally save the image out with bounding boxes overlaid.
  //auto original_image = Identity(root.WithOpName(original_name), image_reader);

  auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

  // Now cast the image data to float so we can do normal math on it.
 // auto float_caster = Cast(root.WithOpName("float_caster"), original_image,
 //                          tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  //auto dims_expander = ExpandDims(root, float_caster, 0);
 auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

  // Bilinearly resize the image to fit the required dimensions.
  /*auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.*/
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
  return Status::OK();
}



// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status TensorFlowObjectDetection::LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}






void TensorFlowObjectDetection::run_tf_object_detection(Mat& frame){
	
    string image = "current_frame.png";
    imwrite(image, frame);
    string image_out = "image_out.png";
    string image_path = tensorflow::io::JoinPath(root_dir_, image);

    Status read_tensor_status =
        ReadTensorFromImageFile(image_path, input_height_, input_width_, input_mean_,
                              input_std_, &image_tensors_);
    if (!read_tensor_status.ok()) {
        LOG(ERROR) << read_tensor_status;
        return;
    }
    const Tensor& resized_tensor = image_tensors_[0];

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status =
        session_->Run({{input_layer_, resized_tensor}},
                   output_layer_, {}, &outputs);
    
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return;
    }

  

    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    int detectionsCount = (int)(num_detections(0));
    auto boxes = outputs[0].flat_outer_dims<float,3>(); 
    int maxCount = min(20,detectionsCount);
    for(int i = 0; i < maxCount; i++){
        if(scores(i) > confidenceThreshold_) {
            float boxClass = classes(i);
            
            float x1 = float(frame.size().width) * boxes(0,i,1);
            float y1 = float(frame.size().height) * boxes(0,i,0);
            
            float x2 = float(frame.size().width) * boxes(0,i,3);
            float y2 = float(frame.size().height) * boxes(0,i,2);
            
            std::ostringstream label;
            label << coco_classes_[int(classes(i))-1] << ", confidence: " << (scores(i)  * 100) << "%";
            std::cout << "Detection " << (i+1) << ": class: " << boxClass << " " <<  label.str()  << std::endl;

            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(255, 255, 255));
            cv::putText(frame, label.str(), cv::Point(x1, y1), 1, 1.0, Scalar(0,0,0));
        }

    }


}