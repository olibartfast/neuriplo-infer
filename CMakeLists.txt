cmake_minimum_required(VERSION 3.20)
project(object-detection-inference)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

# Define the default backend if not set from the command line
if(NOT DEFINED DEFAULT_BACKEND)
    unset(DEFAULT_BACKEND CACHE)
    set(DEFAULT_BACKEND "ONNX_RUNTIME" CACHE STRING "Default backend for inference" FORCE)
endif()

message(STATUS "Home path: $ENV{HOME}")

# Fetch the VideoCapture project from GitHub
include(FetchContent)
FetchContent_Declare(
  VideoCapture
  GIT_REPOSITORY https://github.com/olibartfast/VideoCapture.git
  GIT_TAG        master  # or the specific tag/branch you want to use
)
FetchContent_MakeAvailable(VideoCapture)
message(STATUS "VideoCapture_SOURCE_DIR module path: ${VideoCapture_SOURCE_DIR}")


# Fetch the InferenceEngines project from GitHub
include(FetchContent)
FetchContent_Declare(
    InferenceEngines
  GIT_REPOSITORY https://github.com/olibartfast/inference-engines.git
  GIT_TAG        master  # or the specific tag/branch you want to use
)
FetchContent_MakeAvailable(InferenceEngines)
message(STATUS "InferenceEngines_SOURCE_DIR module path: ${InferenceEngines_SOURCE_DIR}")

set(DETECTORS_ROOT ${CMAKE_CURRENT_LIST_DIR}/detectors)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
message(STATUS "Cmake module path: ${CMAKE_MODULE_PATH}")

# Add subdirectories
add_subdirectory(${DETECTORS_ROOT})

include(AddCompileDefinitions)

# Main executable
set(SOURCES main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories and link libraries
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/detectors/src
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${VideoCapture_SOURCE_DIR}/include 
    ${VideoCapture_SOURCE_DIR}/src
    ${InferenceEngines_SOURCE_DIR}/include
    ${InferenceEngines_SOURCE_DIR}/backends
    ${InferenceEngines_SOURCE_DIR}/backends/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glog::glog
    ${OpenCV_LIBS}
    DetectorsLib
    InferenceEngines
    VideoCapture
)

include(LinkBackend)
include(SetCompilerFlags)