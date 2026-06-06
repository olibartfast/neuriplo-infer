cmake_minimum_required(VERSION 3.24)

# Read version from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_RAW)
string(STRIP "${PROJECT_VERSION_RAW}" PROJECT_VERSION_RAW)
# Extract semver (strip -dev or other suffixes)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PROJECT_SEMVER "${PROJECT_VERSION_RAW}")
if(PROJECT_SEMVER STREQUAL "")
    message(FATAL_ERROR
        "VERSION file '${CMAKE_CURRENT_SOURCE_DIR}/VERSION' does not contain a valid "
        "semantic version (expected MAJOR.MINOR.PATCH). Got: '${PROJECT_VERSION_RAW}'")
endif()

project(neuriplo-infer VERSION ${PROJECT_SEMVER})

include(cmake/CompileSpeed.cmake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PROJECT_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
)

option(WERROR "Treat compiler warnings as errors" OFF)
if(WERROR)
    list(APPEND PROJECT_WARNING_FLAGS -Werror)
endif()

option(ENABLE_APP_TESTS "Enable unit testing for app module" OFF)
if(ENABLE_APP_TESTS)
    enable_testing()
endif()

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

include(FetchContent)

FetchContent_Declare(
    neuriplo-tasks
    GIT_REPOSITORY https://github.com/olibartfast/neuriplo-tasks.git
    GIT_TAG        ${NEURIPLO_TASKS_VERSION}  # Use specific version instead of master
)

# Fetch the neuriplo project from GitHub with specific version
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

# Disable subproject tests and sample apps when fetched as dependencies.
set(BUILD_INFERENCE_ENGINE_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Prefer a sibling checkout during local cross-repo development.
set(_NEURIPLO_TASKS_SIBLING "${CMAKE_CURRENT_SOURCE_DIR}/../neuriplo-tasks")
if(NOT EXISTS "${_NEURIPLO_TASKS_SIBLING}/CMakeLists.txt")
    set(_NEURIPLO_TASKS_SIBLING "${CMAKE_CURRENT_SOURCE_DIR}/../vision-core")
endif()
if(EXISTS "${_NEURIPLO_TASKS_SIBLING}/CMakeLists.txt")
    set(FETCHCONTENT_SOURCE_DIR_NEURIPLO-TASKS "${_NEURIPLO_TASKS_SIBLING}" CACHE PATH "" FORCE)
endif()

FetchContent_MakeAvailable(neuriplo VideoCapture neuriplo-tasks)
message(STATUS "neuriplo-tasks_SOURCE_DIR: ${neuriplo-tasks_SOURCE_DIR}")
message(STATUS "neuriplo_SOURCE_DIR: ${neuriplo_SOURCE_DIR}")
message(STATUS "VideoCapture_SOURCE_DIR: ${VideoCapture_SOURCE_DIR}")

# Configure VideoCapture target with OpenCV after it's been fetched
if(TARGET VideoCapture)
    target_include_directories(VideoCapture PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(VideoCapture PRIVATE ${OpenCV_LIBS})
endif()

# Validate all dependencies after they have been fetched
validate_project_dependencies()

# Define paths
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")

    # Add the app module subdirectory
    add_subdirectory(app)

# Print final configuration summary
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Backend: ${DEFAULT_BACKEND}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

message(STATUS "=====================================")
