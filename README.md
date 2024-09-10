### Object Detection Inference
* Inference for object detection from a video or image input source, with support for multiple switchable frameworks to manage the inference process, and optional GStreamer integration for video capture.
## Dependencies (In parentheses, version tested in this project)
## Required
* CMake 
* OpenCV (apt install libopencv-dev)
* glog (apt install libgoogle-glog-dev)
* C++ compiler with C++17 support (i.e. GCC 8.0 and later)

* One of the following Inference Backend, wrapped in [Inference Engines Library](https://github.com/olibartfast/inference-engines):
    * OpenCV DNN Moduke
    * ONNX Runtime (1.15.1 gpu package)
    * LibTorch (2.0.1-cu118)
    * TensorRT (8.6.1.6)
    * OpenVino (2023.2) 
    * Libtensorflow (2.13)

### Optional 
* GStreamer (1.20.3), wrapped in [VideoCapture Library](https://github.com/olibartfast/videocapture)
* CUDA (if you want to use GPU, CUDA 12 is supported for LibTorch and TensorRT, I used CUDA 11.8 for onnx-rt)

### Notes
 - If you need a specific inference backend, set DEFAULT_BACKEND in CMakeLists with the appropriate option (i.e. ONNX_RUNTIME, LIBTORCH, TENSORRT, LIBTENSORFLOW, OPENCV_DNN, OPENVINO) or set it using cmake from the command line. If no inference backend is specified, the OpenCV-DNN module will be used by default.
- Models with dynamic axis are currently not supported(at least not all)
- Windows build not supported.



## To build and compile  
```
mkdir build
cd build
cmake -DDEFAULT_BACKEND=chosen_backend -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

To enable GStreamer support, you can add -DUSE_GSTREAMER=ON when running cmake, like this:
```
mkdir build
cd build
cmake -DDEFAULT_BACKEND=chosen_backend -DUSE_GSTREAMER=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

This will set the USE_GSTREAMER option to "ON" during the CMake configuration process, enabling GStreamer support in your project.  
Remember to replace chosen_backend with your actual backend selection.
## Usage
```
./object-detection-inference \
    --type=<model type> \
    --source="rtsp://cameraip:port/somelivefeed" (or --source="path/to/video.format") (or --source="path/to/image.format") \
    --labels=</path/to/labels/file> \
    --weights=<path/to/model/weights> [--config=</path/to/model/config>] [--min_confidence=<confidence value>] [--use-gpu] [--warmup] [--benchmark]
```

### Parameters

- `--type=<model type>`: Specifies the type of object detection model to use. Possible values include `yolov4`, `yolov5`, `yolov6`, `yolov7`, `yolov8`, `yolov9`,  `yolov10`, `rtdetr`, and `rtdetrul`. Choose the appropriate model based on your requirements.

- `--source=<source>`: Defines the input source for the object detection. It can be:
  - A live feed URL, e.g., `rtsp://cameraip:port/somelivefeed`
  - A path to a video file, e.g., `path/to/video.format`
  - A path to an image file, e.g., `path/to/image.format`

- `--labels=<path/to/labels/file>`: Specifies the path to the file containing the class labels. This file should list the labels used by the model, each label on a new line.

- `--weights=<path/to/model/weights>`: Defines the path to the file containing the model weights. This file is essential for the model to perform inference.

- `[--config=<path/to/model/config>]`: (Optional) Specifies the path to the model configuration file. This file contains the model architecture and other configurations necessary for setting up the inference. This parameter is primarily needed if the model is from the OpenVINO backend.

- `[--min_confidence=<confidence value>]`: (Optional) Sets the minimum confidence threshold for detections. Detections with a confidence score below this value will be discarded. The default value is `0.25`.

- `[--use-gpu]`: (Optional) Activates GPU support for inference. This can significantly speed up the inference process if a compatible GPU is available.

- `[--warmup]`: (Optional) Enables GPU warmup. Warming up the GPU before performing actual inference can help achieve more consistent and optimized performance. This parameter is relevant only if the inference is being performed on an image source.

- `[--benchmark]`: (Optional) Enables benchmarking mode. In this mode, the application will run multiple iterations of inference to measure and report the average inference time. This is useful for evaluating the performance of the model and the inference setup. This parameter is relevant only if the inference is being performed on an image source.

### To check all available options:
```
./object-detection-inference --help
```
### Run the demo example:
Running inference with yolov8s and the TensorRT backend:  
build setting for cmake DEFAULT_BACKEND=TENSORRT, then run
```
./object-detection-inference \
    --type=yolov8 \
    --weights=/path/to/weights/your_yolov8s.engine \
    --source=/path/to/video.mp4 \
    --labels=/path/to/labels.names
```

Run the inference with rtdetr-l and the Onnx runtime backend:  
build setting for cmake DEFAULT_BACKEND=ONNX_RUNTIME, then run
```
./object-detection-inference  \
    --type=rtdetr \
    --weights=/path/to/weights/your_rtdetr-l.onnx \
    --source=/path/to/video.mp4 \
    --labels=/path/to/labels.names [--use-gpu]
```


## Run with Docker
### Building the Docker Image
* Inside the project, in the [Dockerfiles folder](docker), there will be a dockerfile for each inference backend (currently onnxruntime, libtorch, tensorrt, openvino)

```bash
docker build --rm -t object-detection-inference:<backend_tag> -f docker/Dockerfile.backend .
```

This command will create a docker image based on the provided docker file.

### Running the Docker Container

Replace the wildcards with your desired options and paths:
```bash
docker run --rm -v<path_host_data_folder>:/app/data -v<path_host_weights_folder>:/weights -v<path_host_labels_folder>:/labels object-detection-inference:<backend_tag> --type=<model_type> --weights=<weight_according_your_backend> --source=/app/data/<image_or_video> --labels=/labels/<labels_file>.
```

 ## Available models

* The following table provides information about available object recognition models and supported framework backends: 
[Link to Table Page](docs/TablePage.md#table-of-models)

 ## Exporting a Model for Inference
 * The following page provides information on how to export supported object recognition models: 
[Link to Export Page](docs/ExportInstructions.md)
* [YOLOv5](docs/ExportInstructions.md#yolov5)
* [YOLOv6](docs/ExportInstructions.md#yolov6)
* [YOLOv7](docs/ExportInstructions.md#yolov7)
* [YOLOv8](docs/ExportInstructions.md#yolov8)
* [YOLOv9](docs/ExportInstructions.md#yolov9)
* [YOLOv10](docs/ExportInstructions.md#yolov10)
* [YOLO-NAS](docs/ExportInstructions.md#yolonas)
* [RT-DETR](docs/ExportInstructions.md#rt-detr-lyuwenyu)
* [RT-DETR (Ultralytics implementation)](docs/ExportInstructions.md#rt-detr-ultralytics)

## References
* [Object detection using the opencv dnn module](https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp)
* [TensorRTx](https://github.com/wang-xinyu/tensorrtx)
* [rtdetr-onnxruntime-deploy](https://github.com/CVHub520/rtdetr-onnxruntime-deploy)

## TO DO
- Add Windows building support
- Some benchmarks
- Object detection models from the Torchvision API (if can be exported to C++ deploy i.e. libtorch/torchscript etc...)

## Feedback
- Any feedback is greatly appreciated, if you have any suggestions, bug reports or questions don't hesitate to open an [issue](https://github.com/olibartfast/object-detection-inference/issues).
