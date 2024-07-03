set(OPENCV_DNN_SOURCES
${INFER_ROOT}/src/opencv-dnn/OCVDNNInfer.cpp
# Add more OpenCV DNN source files here if needed
)

list(APPEND SOURCES ${OPENCV_DNN_SOURCES})

add_compile_definitions(USE_OPENCV_DNN)