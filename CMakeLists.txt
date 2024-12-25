cmake_minimum_required(VERSION 3.20)
project(object-detection-inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

# Set DEFAULT_BACKEND as a CACHE variable with its type and a help string.
if(NOT DEFINED DEFAULT_BACKEND)
   set(DEFAULT_BACKEND "ONNX_RUNTIME" CACHE STRING "Default inference backend: OPENCV_DNN, ONNX_RUNTIME, LIBTORCH, TENSORRT, OPENVINO, LIBTENSORFLOW")
endif()

# or set manually in cache to test other backend:
# set(DEFAULT_BACKEND "LIBTORCH" CACHE STRING "Default inference backend" FORCE)
# set(USE_GSTREAMER ON)

message(STATUS "Home path: $ENV{HOME}")

# Fetch the InferenceEngines project from GitHub
include(FetchContent)
FetchContent_Declare(
    InferenceEngines
    GIT_REPOSITORY https://github.com/olibartfast/inference-engines.git
    GIT_TAG        feature/model_info  # or specify the exact tag/branch as needed
)

FetchContent_MakeAvailable(InferenceEngines)
message(STATUS "InferenceEngines_SOURCE_DIR: ${InferenceEngines_SOURCE_DIR}")

# Define paths
set(DETECTORS_ROOT ${CMAKE_SOURCE_DIR}/detectors)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")

# Add subdirectories for the detectors module
add_subdirectory(${DETECTORS_ROOT})

# Option to build only the detectors library
option(BUILD_ONLY_LIB "Build only the detectors library" OFF)

if(NOT BUILD_ONLY_LIB)
    # Add the app module subdirectory
    add_subdirectory(app)
endif()

# Option to enable unit tests for the detectors library
option(ENABLE_DETECTORS_TESTS "Enable unit testing for detectors lib" OFF)

if(ENABLE_DETECTORS_TESTS)
    enable_testing()
    add_subdirectory(${DETECTORS_ROOT}/test)
endif()