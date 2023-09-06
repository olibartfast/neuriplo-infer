### Object detection inference
* Inference for object detection from an IP camera or video stream using GStreamer for video capture, along with the integration of multiple switchable frameworks to manage the inference process.
##  Dependencies (In parentheses version used in this project)
### Required
* CMake (3.22.1)
* GStreamer (1.20.3)
* OpenCV (4.7.0) 
* C++ compiler with C++17 support (i.e. GCC 8.0 and newer)
### Optionally 
* Tensorflow prebuilt library from [Institut für Kraftfahrzeuge](https://github.com/ika-rwth-aachen/libtensorflow_cc)
* CUDA (if you want to use GPU, CUDA 12 is supported for LibTorch and TensorRT, I used CUDA 11.8 for onnx-rt)
* ONNX Runtime (1.15.1 gpu package)
* LibTorch (2.0.1-cu118)
* TensorRT (8.6.1.6)
### Notes
 If needed specific inference backend, set DEFAULT_BACKEND in CMakeLists with proper option(i.e  ONNX_RUNTIME, LIBTORCH, TENSORRT, LIBTENSORFLOW, OPENCV_DNN) or set it using cmake from command line. If not inference backend is specified OpenCV-DNN module is used as default 


## To Build and Compile  
* mkdir build
* cd build
* cmake -DDEFAULT_BACKEND=chosen_backend -DCMAKE_BUILD_TYPE=Release .. 
* cmake --build .

## Usage
```
./object-detection-inference --type=<Model Type> --link="rtsp://cameraip:port/somelivefeed" (or --link="path/to/video.format") --labels=</path/to/labels/file>  --weights=<path/to/model/weights> [--conf=</path/to/model/config>] [--min_confidence=<Conf Value>]
``` 
### To check all available options:
```
./object-detection-inference --help
```
### Run demo example:
Run inference using yolov8s and TensorRT backend:  
build setting for cmake DEFAULT_BACKEND=TENSORRT, then launch
```
./object-detection-inference --type=yolov8 --weights=/path/to/weights/your_yolov8s.engine --link=/path/to/video.mp4 --labels=/path/to/labels.names
```

Run inference using rtdetr-l and Onnx-runtime backend:  
build setting for cmake DEFAULT_BACKEND=ONNX_RUNTIME, then launch
```
./object-detection-inference --type=rtdetr --weights=/path/to/weights/your_rtdetr-l.onnx --link=/path/to/video.mp4 --labels=/path/to/labels.names
```

## Available Models

* The table below provides information about available models for object detection and supported framework backends: 
[Link to Table Page](TablePage.md#table-of-models)


 ### Export a model for the inference
* [YoloV8](ExportInstructions.md#yolov8)
* [YoloNas](ExportInstructions.md#yolonas)
* [YoloV5](ExportInstructions.md#yolov5)
* [YoloV6](ExportInstructions.md#yolov6)
* [YoloV7](ExportInstructions.md#yolov7)
* [RT-DETR](ExportInstructions.md#RT-DETR)

## References
* Using GStreamer to receive a video stream and process it with OpenCV:  
https://stackoverflow.com/questions/10403588/adding-opencv-processing-to-gstreamer-application 


* Object detection using dnn module:  
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp  





