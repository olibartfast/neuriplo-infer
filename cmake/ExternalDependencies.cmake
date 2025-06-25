# ExternalProject-based dependency management
# This module provides an alternative to manual download scripts using CMake's ExternalProject

include(ExternalProject)

# Function to setup ONNX Runtime using ExternalProject
function(setup_onnx_runtime_external)
    if(DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
        message(STATUS "Setting up ONNX Runtime using ExternalProject...")
        
        ExternalProject_Add(
            onnxruntime_external
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}.tgz
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            PREFIX ${CMAKE_BINARY_DIR}/external/onnxruntime
            DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/external/onnxruntime/download
            SOURCE_DIR ${CMAKE_BINARY_DIR}/external/onnxruntime/src
            BINARY_DIR ${CMAKE_BINARY_DIR}/external/onnxruntime/build
            INSTALL_DIR ${CMAKE_BINARY_DIR}/external/onnxruntime/install
        )
        
        # Set the ONNX Runtime directory to the extracted location
        set(ONNX_RUNTIME_DIR ${CMAKE_BINARY_DIR}/external/onnxruntime/src/onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION} PARENT_SCOPE)
        
        message(STATUS "ONNX Runtime will be available at: ${ONNX_RUNTIME_DIR}")
    endif()
endfunction()

# Function to setup LibTorch using ExternalProject
function(setup_libtorch_external)
    if(DEFAULT_BACKEND STREQUAL "LIBTORCH")
        message(STATUS "Setting up LibTorch using ExternalProject...")
        
        # Determine compute platform
        if(DEFINED COMPUTE_PLATFORM)
            set(compute_platform ${COMPUTE_PLATFORM})
        else()
            set(compute_platform "cpu")
        endif()
        
        ExternalProject_Add(
            libtorch_external
            URL https://download.pytorch.org/libtorch/${compute_platform}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+${compute_platform}.zip
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            PREFIX ${CMAKE_BINARY_DIR}/external/libtorch
            DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/external/libtorch/download
            SOURCE_DIR ${CMAKE_BINARY_DIR}/external/libtorch/src
            BINARY_DIR ${CMAKE_BINARY_DIR}/external/libtorch/build
            INSTALL_DIR ${CMAKE_BINARY_DIR}/external/libtorch/install
        )
        
        # Set the LibTorch directory to the extracted location
        set(LIBTORCH_DIR ${CMAKE_BINARY_DIR}/external/libtorch/src/libtorch PARENT_SCOPE)
        
        message(STATUS "LibTorch will be available at: ${LIBTORCH_DIR}")
    endif()
endfunction()

# Function to setup TensorRT using ExternalProject (if publicly available)
function(setup_tensorrt_external)
    if(DEFAULT_BACKEND STREQUAL "TENSORRT")
        message(STATUS "Setting up TensorRT using ExternalProject...")
        
        # Note: TensorRT requires NVIDIA developer account, so this might not work
        # for all users. Fallback to manual installation is recommended.
        
        ExternalProject_Add(
            tensorrt_external
            URL https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            PREFIX ${CMAKE_BINARY_DIR}/external/tensorrt
            DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/external/tensorrt/download
            SOURCE_DIR ${CMAKE_BINARY_DIR}/external/tensorrt/src
            BINARY_DIR ${CMAKE_BINARY_DIR}/external/tensorrt/build
            INSTALL_DIR ${CMAKE_BINARY_DIR}/external/tensorrt/install
        )
        
        # Set the TensorRT directory to the extracted location
        set(TENSORRT_DIR ${CMAKE_BINARY_DIR}/external/tensorrt/src/TensorRT-${TENSORRT_VERSION} PARENT_SCOPE)
        
        message(STATUS "TensorRT will be available at: ${TENSORRT_DIR}")
    endif()
endfunction()

# Function to create dependency targets
function(create_dependency_targets)
    if(DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
        # Create a target for ONNX Runtime
        add_library(onnxruntime_external_lib INTERFACE)
        add_dependencies(onnxruntime_external_lib onnxruntime_external)
        target_include_directories(onnxruntime_external_lib INTERFACE ${ONNX_RUNTIME_DIR}/include)
        target_link_directories(onnxruntime_external_lib INTERFACE ${ONNX_RUNTIME_DIR}/lib)
        target_link_libraries(onnxruntime_external_lib INTERFACE ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so)
    endif()
    
    if(DEFAULT_BACKEND STREQUAL "LIBTORCH")
        # Create a target for LibTorch
        add_library(libtorch_external_lib INTERFACE)
        add_dependencies(libtorch_external_lib libtorch_external)
        target_include_directories(libtorch_external_lib INTERFACE ${LIBTORCH_DIR}/include)
        target_link_directories(libtorch_external_lib INTERFACE ${LIBTORCH_DIR}/lib)
        target_link_libraries(libtorch_external_lib INTERFACE ${LIBTORCH_DIR}/lib/libtorch.so)
    endif()
    
    if(DEFAULT_BACKEND STREQUAL "TENSORRT")
        # Create a target for TensorRT
        add_library(tensorrt_external_lib INTERFACE)
        add_dependencies(tensorrt_external_lib tensorrt_external)
        target_include_directories(tensorrt_external_lib INTERFACE ${TENSORRT_DIR}/include)
        target_link_directories(tensorrt_external_lib INTERFACE ${TENSORRT_DIR}/lib)
        target_link_libraries(tensorrt_external_lib INTERFACE nvinfer nvonnxparser)
    endif()
endfunction()

# Function to setup all external dependencies
function(setup_all_external_dependencies)
    message(STATUS "Setting up external dependencies using ExternalProject...")
    
    setup_onnx_runtime_external()
    setup_libtorch_external()
    setup_tensorrt_external()
    
    # Create dependency targets
    create_dependency_targets()
    
    message(STATUS "External dependencies setup completed")
endfunction() 