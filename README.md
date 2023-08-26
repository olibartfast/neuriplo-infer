### Object detection inference from IP camera RTSP and video stream using GStreamer HW acceleration and varius inference engine backends

##  Dependencies (In parentheses version used in this project)
### Required
* CMake (3.22.1)
* GStreamer (1.20.3)
* OpenCV (4.7.0) 
* C++ compiler with C++17 support
### Optionally 
* Tensorflow prebuilt library from [Institut für Kraftfahrzeuge](https://github.com/ika-rwth-aachen/libtensorflow_cc)
* CUDA (if you want to use GPU, CUDA 12 is supported for LibTorch and TensorRT, I used CUDA 11.8 for onnx-rt)
* ONNX Runtime (1.15.1 gpu package)
* LibTorch (2.0.1-cu118)
* TensorRT (8.6.1.6)
### Notes
 If needed specific inference backend, set DEFAULT_BACKEND in CMakeLists with proper option or set it using cmake from command line. If not inference backend is specified OpenCV-DNN module is used as default 

 ### Export the model for the inference
 [YoloV8](ExportInstructions.md#yolov8)

## To Build and Compile  
* mkdir build
* cd build
* cmake ..
* make

## Usage
```
./object-detection-inference --type=<Model Type> --link="rtsp://cameraip:port/somelivefeed" (or --link="path/to/video.format") --labels=</path/to/labels/file>  --weights=<path/to/model/weights> [--conf=</path/to/model/config>] [--min_confidence=<Conf Value>]
``` 
### To check all available options:
```
./object-detection-inference --help
```

> **Note:** The table below provides information about models for object detection and supported framework backends. 

> Available Models:
> - YOLOv4/YOLOv4-tiny
> - YOLOv7x/YOLOv7-tiny
> - YOLOv5n/s/m/l/x
> - YOLOv6n/s/m/l
> - YOLOv8n/s/m/l/x
> - YOLO-NAS-s/m/l/x
> - TensorFlow Object Detection API

[Link to Table](TablePage.md#table-of-models)


## References
* Using GStreamer to receive a video stream and process it with OpenCV:  
https://stackoverflow.com/questions/10403588/adding-opencv-processing-to-gstreamer-application 


* Object detection using dnn module:  
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp  





