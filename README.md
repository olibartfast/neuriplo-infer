# People and object detection from IP camera RTSP video stream 



Using GStreamer and OpenCV libraries combining the code from:   
https://github.com/opencv/opencv/blob/master/samples/cpp/peopledetect.cpp for HoG + SVM detector  
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp for object detection using dnn module    
and Mikael Lepistö answer at:  
https://stackoverflow.com/questions/10403588/adding-opencv-processing-to-gstreamer-application  


To train your own HoG detector use:  
https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

##  Dependencies
GStreamer 1.0 and OpenCV 3.3.1

## Compilation  
make  
## running object detection with Mobilenet SSD using Caffe framework
./detector --arch=mobilenet --min_confidence=CONF_VALUE(for example 0.6) --link="rtsp://cameraip:port/somelivefeed"    
caffemodel and prototxt for deploying(download inside models folder): https://github.com/chuanqi305/MobileNet-SSD

## running with HoG + SVM People Detector 
./detector --arch=svm --link="rtsp://cameraip:port/somelivefeed"

## running object detection with Yolo
./detector --arch=yolov2(or yolov2-tiny) --min_confidence=0.6 --link="rtsp://cameraip:port/somelivefeed"  
weigths and cfg files to download inside models folder from https://pjreddie.com/darknet/yolo/  

## running people multibox detector with TensorFlow (1.8)
./detector --arch=tf-multibox-detector --link="rtsp://cameraip:port/somelivefeed"  
Multibox detector code based on:  
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/multibox_detector/main.cc  

## running object detection with TensorFlow
./detector --arch=tf-object-detector --min_confidence=0.6 --link="rtsp://cameraip:port/livefeed" 
### Object detection code based on:
https://github.com/tensorflow/models/issues/1741  and https://github.com/moorage/OpenCVTensorflowExample  
### other useful links:
https://github.com/lysukhin/tensorflow-object-detection-cpp  
https://medium.com/@fanzongshaoxing/tensorflow-c-api-to-run-a-object-detection-model-4d5928893b02  

### Tensorflow detection model zoo(to download inside models folder):  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md


### To build Tensorflow shared library with Bazel follow:  
https://tuatini.me/building-tensorflow-as-a-standalone-project/




Tested with Sricam SP009 720P camera   
