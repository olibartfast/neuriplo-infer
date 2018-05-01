DNN = 1
DEBUG = 1

CFLAGS=-Wall -Wfatal-errors 

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(DNN), 1) 
COMMON+= -DDNN
CFLAGS+= -DDNN
endif


all:
	g++ -o detector  main.cpp HogSvmDetector.cpp DnnDetector.cpp $(COMMON) $(CFLAGS)  `pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 opencv`
clean:
	rm detector 
