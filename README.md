# People detection from IP camera stream



Using GStreamer and OpenCV libraries and combining the code from:   
https://github.com/opencv/opencv/blob/master/samples/cpp/peopledetect.cpp   
and Mikael Lepistö answer at:  
https://stackoverflow.com/questions/10403588/adding-opencv-processing-to-gstreamer-application  


##  Dependencies
GStreamer 1.0 and OpenCV (used 3.3.1)

## Compilation and running 
make  
./detector --link="rtsp://cameraip:port/somelivefeed"

