# People detection from IP camera stream (in progress)



Using GStreamer and OpenCV libraries combining the code from:   
https://github.com/opencv/opencv/blob/master/samples/cpp/peopledetect.cpp for HoG + SVM detector  
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp for DNN detector  
and Mikael Lepistö answer at:  
https://stackoverflow.com/questions/10403588/adding-opencv-processing-to-gstreamer-application  

To train your own detector use:  
https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

##  Dependencies
GStreamer 1.0 and OpenCV (used version 3.3.1)

## Compilation and running 
make  
./detector --link="rtsp://cameraip:port/somelivefeed"

