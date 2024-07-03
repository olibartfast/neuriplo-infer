cmake_minimum_required(VERSION 3.20)
project(object-detection-inference)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

message(STATUS "Home path: $ENV{HOME}")


# Define the default backend if not set from the command line
if(NOT DEFINED DEFAULT_BACKEND)
    unset(DEFAULT_BACKEND CACHE)
    set(DEFAULT_BACKEND "LIBTORCH" CACHE STRING "Default backend for inference" FORCE)
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

# Fetch the VideoCapture project from GitHub
include(FetchContent)
FetchContent_Declare(
  VideoCaptureLib
  GIT_REPOSITORY https://github.com/olibartfast/VideoCapture.git
  GIT_TAG        master  # or the specific tag/branch you want to use
)

FetchContent_MakeAvailable(VideoCaptureLib)


message(STATUS "VideoCaptureLib_SOURCE_DIR module path: ${VideoCaptureLib_SOURCE_DIR}")

set(DETECTORS_ROOT ${CMAKE_CURRENT_LIST_DIR}/components/detectors)
set(INFER_ROOT ${CMAKE_CURRENT_LIST_DIR}/components/inference-engines)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
message(STATUS "Cmake module path: ${CMAKE_MODULE_PATH}")

# Add subdirectories
add_subdirectory(${DETECTORS_ROOT})
add_subdirectory(${INFER_ROOT})

# Main executable
set(SOURCES main.cpp)



include(SelectBackend)

add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories and link libraries
target_include_directories(${PROJECT_NAME} PRIVATE
    inc
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${VideoCaptureLib_SOURCE_DIR}/include 
    ${VideoCaptureLib_SOURCE_DIR}/src
    components/inference-engines/src
    components/detectors/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    spdlog::spdlog_header_only
    ${OpenCV_LIBS}
    DetectorsLib
    InferenceEnginesLib
    VideoCaptureLib
)


include(LinkBackend)
include(SetCompilerFlags)
