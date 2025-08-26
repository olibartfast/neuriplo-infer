if(DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
    add_compile_definitions(USE_OPENCV_DNN)
    message(STATUS "Using OpenCV DNN backend")
    
elseif (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    # ONNX Runtime configuration - managed by neuriplo library
    message(STATUS "Using ONNX Runtime backend (managed by neuriplo)")
    add_compile_definitions(USE_ONNX_RUNTIME)
    
elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    # LibTorch configuration - managed by neuriplo library
    message(STATUS "Using LibTorch backend (managed by neuriplo)")
    add_compile_definitions(USE_LIBTORCH)

    # Enable GLOG for LibTorch
    # https://discuss.pytorch.org/t/libtorch-glog-doesnt-print/63822/3
    # The -DC10_USE_GLOG definition is required because LibTorch (C++ PyTorch)
    # uses its own logging system by default called c10. To use glog instead, you need to:
    # Tell the compiler to use glog instead of the default c10 logging system by defining C10_USE_GLOG    
    add_definitions(-DC10_USE_GLOG)
    
elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    # TensorRT configuration - managed by neuriplo library
    message(STATUS "Using TensorRT backend (managed by neuriplo)")
    add_compile_definitions(USE_TENSORRT)
    
elseif (DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW")
    message(STATUS "Using TensorFlow backend (managed by neuriplo)")
    add_compile_definitions(USE_LIBTENSORFLOW)
    
elseif (DEFAULT_BACKEND STREQUAL "OPENVINO")
    message(STATUS "Using OpenVINO backend (managed by neuriplo)")
    add_compile_definitions(USE_OPENVINO)
    
else()
    message(FATAL_ERROR "Unknown backend: ${DEFAULT_BACKEND}. Supported backends: OPENCV_DNN, ONNX_RUNTIME, LIBTORCH, TENSORRT, OPENVINO, LIBTENSORFLOW")
endif()

# Note: Inference backend version management and path configuration
# should be handled by the neuriplo library, not this project.
# This project only sets compile definitions for the selected backend.