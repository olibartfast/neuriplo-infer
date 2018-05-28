DEBUG = 0

CFLAGS=-Wall -Wfatal-errors 

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g -std=c++11 
else
OPTS=-Ofast -std=c++11 
endif

CFLAGS+=$(OPTS)

COMMON+=`pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 opencv`
COMMON+= -I/opt/tensorflow  -I/opt/tensorflow/bazel-genfiles 
COMMON+= -I/tmp/proto/include/ -I/tmp/eigen/include/eigen3/
LDLIBS+=`pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 opencv`
LDLIBS+= -ltensorflow_cc -ltensorflow_framework 


all:
	g++ -o detector  main.cpp GStreamerOpenCV.cpp Detector.cpp HogSvmDetector.cpp MobileNetSSD.cpp TensorFlowMultiboxDetector.cpp Yolo.cpp $(COMMON) $(CFLAGS) $(LDLIBS)  
clean:
	rm detector 
