# Unset cache compiler definitions for the selected framework
if(DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
    include(OpenCVdnn)
elseif (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    # Set ONNX Runtime
    include(ONNXRuntime)
elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    # Set libtorch
    include(LibTorch)
elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    # Set tensorrt
    include(TensorRT)
elseif (DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW")
    # Set TensorFlow
    include(LibTensorFlow)
endif()

