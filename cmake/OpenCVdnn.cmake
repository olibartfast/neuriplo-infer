set(OPENCV_DNN_SOURCES
src/opencv-dnn/YoloVn.cpp
src/opencv-dnn/YoloNas.cpp
src/opencv-dnn/YoloV8.cpp
src/opencv-dnn/YoloV4.cpp
# Add more OpenCV DNN source files here if needed
)

list(APPEND SOURCES ${OPENCV_DNN_SOURCES})

add_compile_definitions(USE_OPENCV_DNN)