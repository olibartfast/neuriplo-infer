### Object detection inference from IP camera RTSP and video stream using GStreamer HW acceleration and OpenCV

##  Dependencies
* GStreamer 1.18.5-1 and OpenCV 4.5.5 (optionally Tensorflow prebuilt library from [
Institut für Kraftfahrzeuge](https://github.com/ika-rwth-aachen/libtensorflow_cc), switch option to ON in CMakeLists or set it using cmake from command line)

## To Build and Compile  
* mkdir build
* cd build
* cmake ..
* make

## Usage
```
./object-detection-inference --type=<Model Type> --link="rtsp://cameraip:port/somelivefeed" (or --link="path/to/video.format") --labels=</path/to/labels/file>  --weights=<path/to/model/weights> --conf=</path/to/model/config> [--min_confidence=<Conf Value>] 
``` 
### To check all available options:
```
./object-detection-inference --help
```

> **Note:** The table below provides information about different models for object detection. Each row represents a specific model and includes the model name, type, demo command, and additional notes.
>
> Available Models:
> - YOLOv4/YOLOv4-tiny
> - YOLOv5n/s/m/l/x...
> - YOLOv8n/s/m/l/x...
> - TensorFlow Object Detection API

[Link to Table](TablePage.md#table-of-models)


## TO DO
* Add support for inference with onnxruntime, tensorrt, openvino

## References
* Using GStreamer to receive a video stream and process it with OpenCV:  
https://stackoverflow.com/questions/10403588/adding-opencv-processing-to-gstreamer-application 


* Object detection using dnn module:  
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp  





