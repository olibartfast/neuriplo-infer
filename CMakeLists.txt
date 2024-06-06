cmake_minimum_required(VERSION 3.20)
project(object-detection-inference)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

message(STATUS "Home path: $ENV{HOME}")

# Define an option to enable or disable GStreamer support
unset(USE_GSTREAMER CACHE)
option(USE_GSTREAMER "Use GStreamer for video capture (optional)" OFF)

# option(BUILD_TESTS "Build test target" OFF) # TODO
# option(BUILD_BENCHMARKS "Build benchmark target" OFF) # TODO

# Define the default backend if not set from the command line
if(NOT DEFINED DEFAULT_BACKEND)
    unset(DEFAULT_BACKEND CACHE)
    set(DEFAULT_BACKEND "OPENCV_DNN" CACHE STRING "Default backend for inference" FORCE)
endif()

# Define the supported backends
set(SUPPORTED_BACKENDS "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "OPENCV_DNN" "TENSORRT" "OPENVINO")

# Check if the specified backend is supported
list(FIND SUPPORTED_BACKENDS ${DEFAULT_BACKEND} SUPPORTED_BACKEND_INDEX)
if (SUPPORTED_BACKEND_INDEX EQUAL -1)
    message(STATUS "Unsupported default backend: ${DEFAULT_BACKEND}")
    set(DEFAULT_BACKEND "OPENCV_DNN" CACHE STRING "Default backend for inference" FORCE)
endif()

message(STATUS "Default backend: ${DEFAULT_BACKEND}")


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
message(STATUS "Cmake module path: ${CMAKE_MODULE_PATH}")

set(DETECTORS_ROOT src/detectors)
set(DETECTORS_SOURCES 
    ${DETECTORS_ROOT}/Detector.cpp 
    ${DETECTORS_ROOT}/YoloNas.cpp 
    ${DETECTORS_ROOT}/RtDetr.cpp 
    ${DETECTORS_ROOT}/RtDetrUltralytics.cpp 
    ${DETECTORS_ROOT}/YoloV4.cpp 
    ${DETECTORS_ROOT}/YoloVn.cpp
    ${DETECTORS_ROOT}/YOLOv10.cpp
    )

set(SOURCES main.cpp src/inference-engines/InferenceInterface.cpp ${DETECTORS_SOURCES})

# Include GStreamer-related settings and source files if USE_GSTREAMER is ON
if (USE_GSTREAMER)
    include(GStreamer)
endif()

include(SelectBackend)

# if(BUILD_TESTS) # TODO
#     message(STATUS "Test enabled")
#     find_package(GTest REQUIRED)
#     enable_testing()
#     add_subdirectory(tests)  # Create a "tests" subdirectory for your test CMakeLists.txt
# endif()

add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories and link libraries
target_include_directories(${PROJECT_NAME} PRIVATE
    inc
    src
    src/detectors
    src/inference-engines
    src/videocapture
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
)


# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only ${OpenCV_LIBS}
)

# Link against GStreamer libraries if USE_GSTREAMER is ON
if (USE_GSTREAMER)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${GSTREAMER_INCLUDE_DIRS}
        ${GST_APP_INCLUDE_DIRS}
        ${GST_VIDEO_INCLUDE_DIRS}
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${GSTREAMER_LIBRARIES}
        ${GST_APP_LIBRARIES}
        ${GST_VIDEO_LIBRARIES}
    )
endif()

include(LinkBackend)

# Set the appropriate compiler flags
include(SetCompilerFlags)
