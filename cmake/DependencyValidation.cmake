# Dependency validation and setup utilities for InferenceEngines library
# This module provides functions to validate and setup inference backend dependencies

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

# Function to validate a dependency exists
function(validate_dependency dependency_name dependency_path)
    if(NOT EXISTS "${dependency_path}")
        message(FATAL_ERROR "${dependency_name} not found at ${dependency_path}. 
        Please ensure the inference backend is properly installed or run the setup script.")
    endif()
    
    message(STATUS "✓ ${dependency_name} found at ${dependency_path}")
endfunction()

# Function to validate ONNX Runtime
function(validate_onnx_runtime)
    if(DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
        validate_dependency("ONNX Runtime" "${ONNX_RUNTIME_DIR}")
        
        # Check for required files
        set(required_files
            "${ONNX_RUNTIME_DIR}/include/onnxruntime_cxx_api.h"
            "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so"
        )
        
        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "ONNX Runtime installation incomplete. Missing: ${file}")
            endif()
        endforeach()
        
        message(STATUS "✓ ONNX Runtime validation passed")
    endif()
endfunction()

# Function to validate TensorRT
function(validate_tensorrt)
    if(DEFAULT_BACKEND STREQUAL "TENSORRT")
        validate_dependency("TensorRT" "${TENSORRT_DIR}")
        
        # Check for required files
        set(required_files
            "${TENSORRT_DIR}/include/NvInfer.h"
            "${TENSORRT_DIR}/lib/libnvinfer.so"
        )
        
        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "TensorRT installation incomplete. Missing: ${file}")
            endif()
        endforeach()
        
        message(STATUS "✓ TensorRT validation passed")
    endif()
endfunction()

# Function to validate LibTorch
function(validate_libtorch)
    if(DEFAULT_BACKEND STREQUAL "LIBTORCH")
        validate_dependency("LibTorch" "${LIBTORCH_DIR}")
        
        # Check for CMake configuration
        if(NOT EXISTS "${LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
            message(FATAL_ERROR "LibTorch CMake configuration not found. Please ensure LibTorch is properly installed.")
        endif()
        
        message(STATUS "✓ LibTorch validation passed")
    endif()
endfunction()

# Function to validate OpenVINO
function(validate_openvino)
    if(DEFAULT_BACKEND STREQUAL "OPENVINO")
        validate_dependency("OpenVINO" "${OPENVINO_DIR}")
        
        # Check for required files
        set(required_files
            "${OPENVINO_DIR}/runtime/include/openvino/openvino.hpp"
            "${OPENVINO_DIR}/runtime/lib/intel64/libopenvino.so"
        )
        
        foreach(file ${required_files})
            if(NOT EXISTS "${file}")
                message(FATAL_ERROR "OpenVINO installation incomplete. Missing: ${file}")
            endif()
        endforeach()
        
        message(STATUS "✓ OpenVINO validation passed")
    endif()
endfunction()

# Function to validate CUDA (if GPU support is requested)
function(validate_cuda)
    if(DEFAULT_BACKEND STREQUAL "TENSORRT" OR DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "✓ CUDA found: ${CUDA_VERSION}")
        else()
            message(WARNING "CUDA not found. GPU support will be disabled for ${DEFAULT_BACKEND} backend.")
        endif()
    endif()
endfunction()

# Function to validate system dependencies
function(validate_system_dependencies)
    # Validate OpenCV
    find_package(OpenCV REQUIRED)
    if(OpenCV_VERSION VERSION_LESS OPENCV_MIN_VERSION)
        message(FATAL_ERROR "OpenCV version ${OpenCV_VERSION} is too old. Minimum required: ${OPENCV_MIN_VERSION}")
    endif()
    message(STATUS "✓ OpenCV ${OpenCV_VERSION} found")
    
    # Validate glog
    find_package(Glog REQUIRED)
    message(STATUS "✓ glog found")
    
    # Validate CMake version
    if(CMAKE_VERSION VERSION_LESS CMAKE_MIN_VERSION)
        message(FATAL_ERROR "CMake version ${CMAKE_VERSION} is too old. Minimum required: ${CMAKE_MIN_VERSION}")
    endif()
    message(STATUS "✓ CMake ${CMAKE_VERSION} found")
endfunction()

# Function to validate all dependencies
function(validate_all_dependencies)
    message(STATUS "=== Validating InferenceEngines Dependencies ===")
    
    validate_system_dependencies()
    
    # Only validate the selected backend
    if(DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
        validate_opencv_dnn()
    elseif(DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
        validate_onnx_runtime()
    elseif(DEFAULT_BACKEND STREQUAL "TENSORRT")
        validate_tensorrt()
    elseif(DEFAULT_BACKEND STREQUAL "LIBTORCH")
        validate_libtorch()
    elseif(DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW")
        validate_libtensorflow()
    elseif(DEFAULT_BACKEND STREQUAL "OPENVINO")
        validate_openvino()
    else()
        message(FATAL_ERROR "Unknown backend: ${DEFAULT_BACKEND}")
    endif()
    
    # Validate CUDA if needed for the selected backend
    validate_cuda()
    
    message(STATUS "=== All InferenceEngines Dependencies Validated Successfully ===")
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
    message(STATUS "If inference backend dependencies are missing, run the following commands:")
    message(STATUS "")
    
    if(DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
        message(STATUS "  ./scripts/setup_onnx_runtime.sh")
    elseif(DEFAULT_BACKEND STREQUAL "TENSORRT")
        message(STATUS "  ./scripts/setup_tensorrt.sh")
    elseif(DEFAULT_BACKEND STREQUAL "LIBTORCH")
        message(STATUS "  ./scripts/setup_libtorch.sh")
    elseif(DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW")
        message(STATUS "  For LibTensorFlow, please install TensorFlow C++ library manually")
        message(STATUS "  or check the project documentation for installation instructions")
    elseif(DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
        message(STATUS "  OpenCV DNN is included with OpenCV installation")
        message(STATUS "  Ensure OpenCV is installed with DNN module support")
    elseif(DEFAULT_BACKEND STREQUAL "OPENVINO")
        message(STATUS "  ./scripts/setup_openvino.sh")
    endif()
    
    message(STATUS "")
    message(STATUS "Or run the unified setup script:")
    message(STATUS "  ./scripts/setup_dependencies.sh --backend ${DEFAULT_BACKEND}")
    message(STATUS "")
endfunction()

# Function to validate OpenCV DNN
function(validate_opencv_dnn)
    if(DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
        # OpenCV is already validated in validate_system_dependencies()
        # Just check that DNN module is available
        find_package(OpenCV REQUIRED)
        
        # Check if OpenCV was compiled with DNN support
        if(NOT OpenCV_FOUND)
            message(FATAL_ERROR "OpenCV not found. OpenCV DNN backend requires OpenCV.")
        endif()
        
        message(STATUS "✓ OpenCV DNN validation passed")
    endif()
endfunction()

# Function to validate LibTensorFlow
function(validate_libtensorflow)
    if(DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW")
        # Add cmake modules path to find our custom FindTensorFlow.cmake
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
        
        find_package(TensorFlow QUIET)
        if(NOT TensorFlow_FOUND)
            message(FATAL_ERROR "LibTensorFlow not found. Please install TensorFlow C++ library or run the setup script.")
        endif()
        
        # Check for required TensorFlow components
        if(NOT DEFINED TensorFlow_INCLUDE_DIRS OR NOT DEFINED TensorFlow_LIBRARIES)
            message(FATAL_ERROR "LibTensorFlow installation incomplete. Missing include directories or libraries.")
        endif()
        
        message(STATUS "✓ LibTensorFlow validation passed")
    endif()
endfunction()
