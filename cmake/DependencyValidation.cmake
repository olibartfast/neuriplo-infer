# Dependency validation and setup utilities for this project
# This module provides functions to validate and setup dependencies managed by this project

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

# Function to validate a dependency exists
function(validate_dependency dependency_name dependency_path)
    if(NOT EXISTS "${dependency_path}")
        message(FATAL_ERROR "${dependency_name} not found at ${dependency_path}. 
        Please ensure the dependency is properly installed or run the setup script.")
    endif()
    
    message(STATUS "✓ ${dependency_name} found at ${dependency_path}")
endfunction()

# Function to validate system dependencies
function(validate_system_dependencies)
    # OpenCV and glog should already be found before this function is called
    # Just validate versions if they're already found
    if(NOT OpenCV_FOUND)
        find_package(OpenCV REQUIRED)
    endif()
    if(OpenCV_VERSION VERSION_LESS OPENCV_MIN_VERSION)
        message(FATAL_ERROR "OpenCV version ${OpenCV_VERSION} is too old. Minimum required: ${OPENCV_MIN_VERSION}")
    endif()
    message(STATUS "✓ OpenCV ${OpenCV_VERSION} validated")
    
    if(NOT glog_FOUND)
        find_package(glog REQUIRED)
    endif()
    message(STATUS "✓ glog validated")
    
    # Validate CMake version
    if(CMAKE_VERSION VERSION_LESS CMAKE_MIN_VERSION)
        message(FATAL_ERROR "CMake version ${CMAKE_VERSION} is too old. Minimum required: ${CMAKE_MIN_VERSION}")
    endif()
    message(STATUS "✓ CMake ${CMAKE_VERSION} found")
endfunction()

# Function to validate fetched dependencies
function(validate_fetched_dependencies)
    # Validate InferenceEngines library
    if(NOT DEFINED InferenceEngines_SOURCE_DIR)
        message(FATAL_ERROR "InferenceEngines library not found. This should be fetched automatically by CMake.")
    endif()
    message(STATUS "✓ InferenceEngines library found at ${InferenceEngines_SOURCE_DIR}")
    
    # Validate VideoCapture library
    if(NOT DEFINED VideoCapture_SOURCE_DIR)
        message(FATAL_ERROR "VideoCapture library not found. This should be fetched automatically by CMake.")
    endif()
    message(STATUS "✓ VideoCapture library found at ${VideoCapture_SOURCE_DIR}")
endfunction()

# Function to validate all dependencies for this project
function(validate_all_dependencies)
    message(STATUS "=== Validating Project Dependencies ===")
    
    validate_system_dependencies()
    validate_fetched_dependencies()
    
    message(STATUS "=== All Project Dependencies Validated Successfully ===")
endfunction()

# Function to check if we're in a Docker environment
function(is_docker_environment result)
    if(EXISTS "/.dockerenv")
        set(${result} TRUE PARENT_SCOPE)
    else()
        set(${result} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Function to provide helpful setup instructions
function(print_setup_instructions)
    message(STATUS "=== Setup Instructions ===")
    message(STATUS "This project uses the InferenceEngines library for inference backend management.")
    message(STATUS "For inference backend setup, please refer to the InferenceEngines library documentation.")
    message(STATUS "")
    message(STATUS "System dependencies can be installed with:")
    message(STATUS "  sudo apt update && sudo apt install -y libopencv-dev libgoogle-glog-dev")
    message(STATUS "")
endfunction()

# Note: Inference backend validation (ONNX Runtime, TensorRT, LibTorch, etc.)
# should be handled by the InferenceEngines library, not this project.
# This project only validates its own dependencies and the fetched libraries. 
