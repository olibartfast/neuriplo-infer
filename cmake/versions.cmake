# Centralized version management for this project's dependencies
# This file should be the single source of truth for versions managed by this project

# External C++ Libraries (fetched via CMake FetchContent)
set(INFERENCE_ENGINES_VERSION "feature/code_refactoring" CACHE STRING "InferenceEngines library version")
set(VIDEOCAPTURE_VERSION "feature/code_refactoring" CACHE STRING "VideoCapture library version")

# System Dependencies (minimum versions)
set(OPENCV_MIN_VERSION "4.6.0" CACHE STRING "Minimum OpenCV version")
set(GLOG_MIN_VERSION "0.6.0" CACHE STRING "Minimum glog version")
set(CMAKE_MIN_VERSION "3.20" CACHE STRING "Minimum CMake version")

# Print version information for debugging
message(STATUS "=== Project Dependency Versions ===")
message(STATUS "InferenceEngines: ${INFERENCE_ENGINES_VERSION}")
message(STATUS "VideoCapture: ${VIDEOCAPTURE_VERSION}")
message(STATUS "OpenCV Min: ${OPENCV_MIN_VERSION}")
message(STATUS "glog Min: ${GLOG_MIN_VERSION}")
message(STATUS "CMake Min: ${CMAKE_MIN_VERSION}")

# Note: Inference backend versions (ONNX Runtime, TensorRT, LibTorch, etc.)
# are managed by the InferenceEngines library, not this project.
# See the InferenceEngines library for backend-specific version management. 
