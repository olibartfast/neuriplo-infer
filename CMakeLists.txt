cmake_minimum_required(VERSION 3.24)

# Read version from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_RAW)
string(STRIP "${PROJECT_VERSION_RAW}" PROJECT_VERSION_RAW)
# Extract semver (strip -dev or other suffixes)
string(REGEX MATCH "^[0-9]+\.[0-9]+\.[0-9]+" PROJECT_SEMVER "${PROJECT_VERSION_RAW}")
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
message(STATUS "OpenCV ${OpenCV_VERSION} found")
message(STATUS "glog found")

include(FetchContent)

# Build-mode switches (defined before any fetch so they can gate it).
#   ENABLE_KSERVE         -> KServe remote-runtime client (no neuriplo needed).
#   ENABLE_LOCAL_BACKENDS -> local inference engines; this is what fetches and
#                            builds neuriplo (OpenCV/glog/ONNX/TensorRT/...).
# A KServe-only build (LOCAL_BACKENDS OFF, KSERVE ON) fetches neither neuriplo
# nor any external contract library; it uses the app-local contract headers in
# app/inc/contract. See docs/KserveRuntime.md.
option(NEURIPLO_INFER_ENABLE_KSERVE "Build KServe remote-runtime client" ON)
option(NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS
       "Build local inference backends (fetches/builds neuriplo)" ON)
if(NOT NEURIPLO_INFER_ENABLE_KSERVE AND NOT NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS)
    message(FATAL_ERROR
        "At least one of NEURIPLO_INFER_ENABLE_KSERVE / "
        "NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS must be ON.")
endif()

set(NEURIPLO_TASKS_GIT_REPOSITORY "https://github.com/olibartfast/neuriplo-tasks.git")

FetchContent_Declare(
    neuriplo-tasks
    GIT_REPOSITORY ${NEURIPLO_TASKS_GIT_REPOSITORY}
    GIT_TAG        ${NEURIPLO_TASKS_VERSION}
)

# Fetch the neuriplo project (local backends) only when they are enabled.
if(NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS)
    FetchContent_Declare(
        neuriplo
        GIT_REPOSITORY https://github.com/olibartfast/neuriplo.git
        GIT_TAG        ${NEURIPLO_VERSION}  # Use specific version instead of master
    )
endif()

FetchContent_Declare(
    VideoCapture
    GIT_REPOSITORY https://github.com/olibartfast/videocapture.git
    GIT_TAG        ${VIDEOCAPTURE_VERSION}  # Use specific version instead of master
)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# The KServe V2 protocol client is a standalone sibling repo (neuriplo-free).
# Fetched only when the KServe runtime is enabled; the adapter (KserveEngine)
# that bridges it to the neuriplo contract stays in app/.
if(NEURIPLO_INFER_ENABLE_KSERVE)
    FetchContent_Declare(
        neuriplo-kserve-client
        GIT_REPOSITORY https://github.com/olibartfast/neuriplo-kserve-client.git
        GIT_TAG        ${NEURIPLO_KSERVE_CLIENT_VERSION}
    )
endif()

# Disable subproject tests and sample apps when fetched as dependencies.
set(BUILD_INFERENCE_ENGINE_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Prefer a sibling checkout during local cross-repo development.
set(_NEURIPLO_TASKS_SIBLING "")
foreach(_candidate IN ITEMS
    "${CMAKE_CURRENT_SOURCE_DIR}/../neuriplo-tasks")
    if(EXISTS "${_candidate}/CMakeLists.txt")
        set(_NEURIPLO_TASKS_SIBLING "${_candidate}")
        break()
    endif()
endforeach()
if(NOT _NEURIPLO_TASKS_SIBLING STREQUAL "")
    set(FETCHCONTENT_SOURCE_DIR_NEURIPLO-TASKS "${_NEURIPLO_TASKS_SIBLING}" CACHE PATH "" FORCE)
endif()

# Prefer a neuriplo sibling checkout during local cross-repo development.
if(NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS)
    set(_NEURIPLO_SIBLING "")
    foreach(_candidate IN ITEMS
        "${CMAKE_CURRENT_SOURCE_DIR}/../neuriplo")
        if(EXISTS "${_candidate}/CMakeLists.txt")
            set(_NEURIPLO_SIBLING "${_candidate}")
            break()
        endif()
    endforeach()
    if(NOT _NEURIPLO_SIBLING STREQUAL "")
        set(FETCHCONTENT_SOURCE_DIR_NEURIPLO "${_NEURIPLO_SIBLING}" CACHE PATH "" FORCE)
    endif()
endif()

# Prefer a neuriplo-kserve-client sibling checkout during local cross-repo dev.
if(NEURIPLO_INFER_ENABLE_KSERVE)
    set(_KSERVE_CLIENT_SIBLING "")
    foreach(_candidate IN ITEMS
        "${CMAKE_CURRENT_SOURCE_DIR}/../neuriplo-kserve-client")
        if(EXISTS "${_candidate}/CMakeLists.txt")
            set(_KSERVE_CLIENT_SIBLING "${_candidate}")
            break()
        endif()
    endforeach()
    if(NOT _KSERVE_CLIENT_SIBLING STREQUAL "")
        set(FETCHCONTENT_SOURCE_DIR_NEURIPLO-KSERVE-CLIENT "${_KSERVE_CLIENT_SIBLING}" CACHE PATH "" FORCE)
    endif()
endif()

if(NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS)
    FetchContent_MakeAvailable(neuriplo)
endif()
FetchContent_MakeAvailable(VideoCapture neuriplo-tasks nlohmann_json)

# KServe transport knobs. The actual gRPC/Protobuf and OpenSSL detection now
# lives in the neuriplo-kserve-client library; these options just map the
# neuriplo-infer-facing names onto the library's options. The library exports a
# PUBLIC KSERVE_CLIENT_WITH_GRPC define when the gRPC transport is compiled in,
# which the app reads to decide whether to build the gRPC client path.
option(NEURIPLO_INFER_ENABLE_GRPC "Build gRPC KServe client" ON)
if(DEFINED VISION_INFERENCE_ENABLE_GRPC)
    set(NEURIPLO_INFER_ENABLE_GRPC ${VISION_INFERENCE_ENABLE_GRPC})
endif()
option(NEURIPLO_INFER_ENABLE_KSERVE_TLS
       "Enable HTTPS (OpenSSL) for the KServe HTTP client" ON)

if(NEURIPLO_INFER_ENABLE_KSERVE)
    set(KSERVE_CLIENT_ENABLE_GRPC ${NEURIPLO_INFER_ENABLE_GRPC} CACHE BOOL "" FORCE)
    set(KSERVE_CLIENT_ENABLE_TLS ${NEURIPLO_INFER_ENABLE_KSERVE_TLS} CACHE BOOL "" FORCE)
    set(KSERVE_CLIENT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(neuriplo-kserve-client)
    message(STATUS "neuriplo-kserve-client_SOURCE_DIR: ${neuriplo-kserve-client_SOURCE_DIR}")
endif()

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
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "KServe runtime: ${NEURIPLO_INFER_ENABLE_KSERVE}")
message(STATUS "Local backends: ${NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS}")
if(NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS)
    message(STATUS "Backend: ${DEFAULT_BACKEND}")
endif()
message(STATUS "=====================================")
