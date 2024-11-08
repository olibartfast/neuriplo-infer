cmake_minimum_required(VERSION 3.20)
project(object-detection-inference)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

# Define the default backend if not set from the command line
if(DEFINED DEFAULT_BACKEND)
    unset(DEFAULT_BACKEND CACHE)
endif()
set(DEFAULT_BACKEND "OPENCV_DNN" CACHE STRING "Default backend for inference" FORCE) # ONNX_RUNTIME, LIBTORCH, TENSORRT, LIBTENSORFLOW, OPENCV_DNN, OPENVINO

message(STATUS "Home path: $ENV{HOME}")

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
message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")

# Add subdirectories for the detectors module
add_subdirectory(${DETECTORS_ROOT})

include(AddCompileDefinitions)

# Option to build only the detectors library
option(BUILD_ONLY_LIB "Build only the detectors library" ON)

if(NOT BUILD_ONLY_LIB)
    # Fetch the VideoCapture project from GitHub
    include(FetchContent)
    FetchContent_Declare(
    VideoCapture
    GIT_REPOSITORY https://github.com/olibartfast/VideoCapture.git
    GIT_TAG        master  # or the specific tag/branch you want to use
    )
    FetchContent_MakeAvailable(VideoCapture)
    message(STATUS "VideoCapture_SOURCE_DIR module path: ${VideoCapture_SOURCE_DIR}")


    # Main executable
    set(SOURCES main.cpp app/src/ObjectDetectionApp.cpp app/src/utils.cpp app/src/CommandLineParser.cpp)
    add_executable(${PROJECT_NAME} ${SOURCES})

    target_include_directories(${PROJECT_NAME} PUBLIC
    ${VideoCapture_SOURCE_DIR}/include 
        ${InferenceEngines_SOURCE_DIR}/include
    )

    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/app/inc
        ${CMAKE_SOURCE_DIR}/app/src
        ${CMAKE_SOURCE_DIR}/detectors/inc
        ${CMAKE_SOURCE_DIR}/detectors/src
        ${CMAKE_SOURCE_DIR}/detectors/src/models
        ${OpenCV_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
    ${VideoCapture_SOURCE_DIR}/src
        ${InferenceEngines_SOURCE_DIR}/backends
        ${InferenceEngines_SOURCE_DIR}/backends/src
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE
        glog::glog
        ${OpenCV_LIBS}
        detectors
        InferenceEngines
        VideoCapture
    )
include(LinkBackend)
include(SetCompilerFlags)    
endif()

# Option to enable unit tests for app module
option(ENABLE_APP_TESTS "Enable unit testing for app module" OFF)

if(ENABLE_APP_TESTS AND NOT BUILD_ONLY_LIB)
    enable_testing()
    add_subdirectory(app/test)
endif()

# Option to enable unit tests for detectors library
option(ENABLE_DETECTORS_TESTS "Enable unit testing for detectors lib" OFF)

if(ENABLE_DETECTORS_TESTS)
    enable_testing()
    add_subdirectory(detectors/test)
endif()
