all:
	g++ -o detector  main.cpp HogSvmDetector.cpp DnnDetector.cpp `pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 opencv`
clean:
	rm detector 
