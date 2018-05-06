DEBUG = 0

CFLAGS=-Wall -Wfatal-errors 

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g -std=c++11 
else
OPTS=-Ofast -std=c++11 
endif

CFLAGS+=$(OPTS)


all:
	g++ -o detector  main.cpp GStreamerOpenCV.cpp HogSvmDetector.cpp MobileNetSSD.cpp $(COMMON) $(CFLAGS)  `pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 opencv`
clean:
	rm detector 
