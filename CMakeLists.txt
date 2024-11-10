cmake_minimum_required(VERSION 3.20)
project(object-detection-inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

# Define the default backend if not set from the command line
if(DEFINED DEFAULT_BACKEND)
    unset(DEFAULT_BACKEND CACHE)
endif()
set(DEFAULT_BACKEND ${DEFAULT_BACKEND} CACHE STRING "Default backend for inference" FORCE)

message(STATUS "Home path: $ENV{HOME}")

# Fetch the InferenceEngines project from GitHub
include(FetchContent)
FetchContent_Declare(
    InferenceEngines
    GIT_REPOSITORY https://github.com/olibartfast/inference-engines.git
    GIT_TAG        master  # or specify the exact tag/branch as needed
)
FetchContent_MakeAvailable(InferenceEngines)
message(STATUS "InferenceEngines_SOURCE_DIR: ${InferenceEngines_SOURCE_DIR}")

# Define paths
set(DETECTORS_ROOT ${CMAKE_CURRENT_LIST_DIR}/detectors)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")

# Add subdirectories for the detectors module
add_subdirectory(${DETECTORS_ROOT})

# Option to build only the detectors library
option(BUILD_ONLY_LIB "Build only the detectors library" OFF)
unset(BUILD_ONLY_LIB CACHE) # Ensure any previous cache is cleared

if(NOT BUILD_ONLY_LIB)
    # Add the app module subdirectory
    add_subdirectory(app)
endif()

# Option to enable unit tests for the detectors library
option(ENABLE_DETECTORS_TESTS "Enable unit testing for detectors lib" OFF)
unset(ENABLE_DETECTORS_TESTS CACHE) # Clear any previous cache

if(ENABLE_DETECTORS_TESTS)
    enable_testing()
    add_subdirectory(detectors/test)
endif()
