cmake_minimum_required(VERSION 3.20)
project(object-detection-inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include centralized version management
include(cmake/versions.cmake)

# Set DEFAULT_BACKEND as a CACHE variable with its type and a help string.
if(NOT DEFINED DEFAULT_BACKEND)
   set(DEFAULT_BACKEND "OPENCV_DNN" CACHE STRING "Default inference backend: OPENCV_DNN, ONNX_RUNTIME, LIBTORCH, TENSORRT, OPENVINO, LIBTENSORFLOW")
endif()

# Video backend options (managed by VideoCapture library)
# set(USE_GSTREAMER ON)  # Enable GStreamer backend
# set(USE_FFMPEG ON)     # Enable FFmpeg backend (takes priority over GStreamer)
# If both enabled, priority: FFmpeg > GStreamer > OpenCV (default)

message(STATUS "Home path: $ENV{HOME}")

# Include dependency validation
include(cmake/DependencyValidation.cmake)

# Find system dependencies first (before fetching external dependencies)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
message(STATUS "✓ OpenCV ${OpenCV_VERSION} found")
message(STATUS "✓ glog found")

# Fetch the neuriplo project from GitHub with specific version
include(FetchContent)
FetchContent_Declare(
    neuriplo
    GIT_REPOSITORY https://github.com/olibartfast/neuriplo.git
    GIT_TAG        ${NEURIPLO_VERSION}  # Use specific version instead of master
)

FetchContent_Declare(
    VideoCapture
    GIT_REPOSITORY https://github.com/olibartfast/videocapture.git
    GIT_TAG        ${VIDEOCAPTURE_VERSION}  # Use specific version instead of master
)

FetchContent_MakeAvailable(neuriplo VideoCapture)
message(STATUS "neuriplo_SOURCE_DIR: ${neuriplo_SOURCE_DIR}")
message(STATUS "VideoCapture_SOURCE_DIR: ${VideoCapture_SOURCE_DIR}")

# Configure VideoCapture target with OpenCV after it's been fetched
if(TARGET VideoCapture)
    target_include_directories(VideoCapture PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(VideoCapture PRIVATE ${OpenCV_LIBS})
endif()

# Validate all dependencies after they have been fetched
validate_all_dependencies()

# Define paths
set(DETECTORS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/detectors)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
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

# Print final configuration summary
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Backend: ${DEFAULT_BACKEND}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build Only Library: ${BUILD_ONLY_LIB}")
message(STATUS "Enable Detector Tests: ${ENABLE_DETECTORS_TESTS}")
message(STATUS "=====================================")