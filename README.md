### Object detection inference from IP camera RTSP and video stream using GStreamer HW acceleration and OpenCV

Using GStreamer and OpenCV libraries combining the code from:   
https://github.com/opencv/opencv/blob/master/samples/cpp/peopledetect.cpp for HoG + SVM detector  
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp for object detection using dnn module    
and Mikael Lepistö answer at:  
https://stackoverflow.com/questions/10403588/adding-opencv-processing-to-gstreamer-application  


To train your own HoG detector use:  
https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

##  Dependencies
* GStreamer 1.0 and OpenCV 4.2.0

## To Build and Compile  
* mkdir build
* cd build
* cmake ..
* make

## running object detection with Mobilenet SSD using Caffe framework
```
./object-detection-inference --type=mobilenet --min_confidence=CONF_VALUE(for example 0.6) --link="rtsp://cameraip:port/somelivefeed"  
```  
* caffemodel and prototxt for deploying(download inside models folder): https://github.com/chuanqi305/MobileNet-SSD

## running with HoG + SVM People Detector 
```
./object-detection-inference --type=svm --link="rtsp://cameraip:port/somelivefeed"
```

## running object detection with YoloV2
```
./object-detection-inference --type=yolov2(or yolov2-tiny) --min_confidence=0.6 --link="rtsp://cameraip:port/somelivefeed"  
```
* Weigths and cfg files to download inside models folder from https://pjreddie.com/darknet/yolo/  

### To check all available options:
```
./object-detection-inference --help
```

## TO DO
* Planning to restore and update inference on tensorflow models from object detection API
* Inference on pytorch models from torchvision object detection API
* Inference on more recent yolo models
* Factory pattern for detector object creation
* Add support for inference with openvino and tensorrt
