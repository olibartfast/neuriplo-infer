# People and object detection from IP camera RTSP video stream 



Using GStreamer and OpenCV libraries combining the code from:   
https://github.com/opencv/opencv/blob/master/samples/cpp/peopledetect.cpp for HoG + SVM detector  
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp for object detection using dnn module    
and Mikael Lepistö answer at:  
https://stackoverflow.com/questions/10403588/adding-opencv-processing-to-gstreamer-application  


To train your own HoG detector use:  
https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

##  Dependencies
GStreamer 1.0 and OpenCV (used version 3.3.1)

## Compilation  
make  
## running object detection with Mobilenet SSD using Caffe framework
./detector --arch=mobilenet --link="rtsp://cameraip:port/somelivefeed"    
caffemodel and prototxt for deploying(download inside models folder): https://github.com/chuanqi305/MobileNet-SSD

## running with HoG + SVM People Detector 
./detector --arch=svm --link="rtsp://cameraip:port/somelivefeed"

## running object detection with Yolo
./detector --arch=yolov2(or yolov2-tiny) --link="rtsp://cameraip:port/somelivefeed"  
weigths and cfg files to download inside models folder from https://pjreddie.com/darknet/yolo/  

## running object detection with TensorFlow 
./detector --arch=tensorflow --link="rtsp://cameraip:port/somelivefeed"  
detection model and config file:  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Tested with Sricam SP009 720P camera   
