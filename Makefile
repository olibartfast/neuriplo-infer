DNN = 1
DEBUG = 0

CFLAGS=-Wall -Wfatal-errors 

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
else
OPTS=-Ofast
endif

CFLAGS+=$(OPTS)

ifeq ($(DNN), 1) 
CFLAGS+= -DDNN
endif


all:
	g++ -o detector  main.cpp HogSvmDetector.cpp MobileNetSSD.cpp $(COMMON) $(CFLAGS)  `pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 opencv`
clean:
	rm detector 
