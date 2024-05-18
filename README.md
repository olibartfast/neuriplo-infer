### Object Detection Inference
* Inference for object detection from a video or image input source, with support for multiple switchable frameworks to manage the inference process, and optional GStreamer integration for video capture.
## Dependencies (In parentheses, version used in this project)
## Required
* CMake (3.22.1)
* OpenCV (4.7.0) (apt install libopencv-dev)
* spdlog (1:1.9.2+ds-0.2) (apt-get install libspdlog-dev)
* C++ compiler with C++17 support (i.e. GCC 8.0 and later)
### Optional 
* GStreamer (1.20.3) 
* CUDA (if you want to use GPU, CUDA 12 is supported for LibTorch and TensorRT, I used CUDA 11.8 for onnx-rt)
* ONNX Runtime (1.15.1 gpu package)
* LibTorch (2.0.1-cu118)
* TensorRT (8.6.1.6)
* OpenVino (2023.2) 
### Notes
 - If you need a specific inference backend, set DEFAULT_BACKEND in CMakeLists with the appropriate option (i.e. ONNX_RUNTIME, LIBTORCH, TENSORRT, LIBTENSORFLOW, OPENCV_DNN, OPENVINO) or set it using cmake from the command line. If no inference backend is specified, the OpenCV-DNN module will be used by default.
- Models with dynamic axis are currently not supported.


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
    --weights=<path/to/model/weights> [--config=</path/to/model/config>] [--min_confidence=<confidence value>].
``` 
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
* Inside the project, in the [Dockerfiles folder](Dockerfiles], there is a dockerfile for each inference backend (currently only onnxruntime)

```bash
docker build --rm -t object-detection-inference:<backend_tag> -f Dockerfiles/Dockerfile.backend .
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

## References
* [Object detection using the opencv dnn module](https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp)
* [TensorRTx](https://github.com/wang-xinyu/tensorrtx)
* [rtdetr-onnxruntime-deploy](https://github.com/CVHub520/rtdetr-onnxruntime-deploy)

## TO DO
- Reimplement Libtensorflow backend
- Run inside a docker container
- Add tests

## Feedback
- Any feedback is greatly appreciated, if you have any suggestions, bug reports or questions don't hesitate to open an [issue](https://github.com/olibartfast/object-detection-inference/issues).
