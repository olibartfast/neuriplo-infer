### Object detection inference from IP camera RTSP and video stream using GStreamer HW acceleration and OpenCV

##  Dependencies
* GStreamer 1.18.5-1 and OpenCV 4.5.5

## To Build and Compile  
* mkdir build
* cd build
* cmake ..
* make

### Running object detection with yolov4/yolov4-tiny
```
./object-detection-inference --type=yolov4(or yolov4-tiny) --min_confidence=0.6 --link="rtsp://cameraip:port/somelivefeed"  
```
* Weigths and .cfg files to download inside models folder from https://github.com/AlexeyAB/darknet/releases/tag/yolov4 

### Running object detection with yolov5s/m/l/x
```
./object-detection-inference --type=yolov5x(or other versions) --min_confidence=0.6 --link="rtsp://cameraip:port/somelivefeed"  
```
* Weigths to put inside models folder after exporting the pretrained .pt file in onnx format using the script from https://github.com/ultralytics/yolov5/blob/master/export.py

### Running object detection with Mobilenet SSD using Caffe framework
```
./object-detection-inference --type=mobilenet --min_confidence=CONF_VALUE(for example 0.6) --link="rtsp://cameraip:port/somelivefeed"  
```  
* Caffemodel and Prototxt for deploying(download inside models folder): https://github.com/chuanqi305/MobileNet-SSD

### Running with HoG + SVM People Detector 
```
./object-detection-inference --type=svm --link="rtsp://cameraip:port/somelivefeed"
```

### To check all available options:
```
./object-detection-inference --help
```

## TO DO
* Add support for inference with onnxruntime, tensorrt, openvino
* Planning to restore and update inference on tensorflow models from object detection API

## References
* Using GStreamer to receive a video stream and process it with OpenCV:  
https://stackoverflow.com/questions/10403588/adding-opencv-processing-to-gstreamer-application 

*  HoG + SVM detector:   
https://github.com/opencv/opencv/blob/master/samples/cpp/peopledetect.cpp

* Train your own HoG detector:  
https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

* Object detection using dnn module:  
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp  

* Sort algorithm:  
https://github.com/david8862/keras-YOLOv3-model-set/tree/master/tracking/cpp_inference/yoloSort




