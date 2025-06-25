# Include framework-specific source files and libraries
# Note: Inference backend linking is handled by the InferenceEngines library
# This file only includes backend-specific source directories from InferenceEngines

if (DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
    target_include_directories(${PROJECT_NAME} PRIVATE ${InferenceEngines_SOURCE_DIR}/backends/opencv-dnn/src)
elseif (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    target_include_directories(${PROJECT_NAME} PRIVATE ${InferenceEngines_SOURCE_DIR}/backends/onnx-runtime/src)
elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    target_include_directories(${PROJECT_NAME} PRIVATE ${InferenceEngines_SOURCE_DIR}/backends/libtorch/src)
    target_compile_definitions(${PROJECT_NAME}  PRIVATE C10_USE_GLOG)
elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    target_include_directories(${PROJECT_NAME} PRIVATE ${InferenceEngines_SOURCE_DIR}/backends/tensorrt/src)
elseif(DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW" )
    target_include_directories(${PROJECT_NAME} PRIVATE ${InferenceEngines_SOURCE_DIR}/backends/libtensorflow/src)
elseif(DEFAULT_BACKEND STREQUAL "OPENVINO")
    target_include_directories(${PROJECT_NAME} PRIVATE ${InferenceEngines_SOURCE_DIR}/backends/openvino/src)
endif()

# Note: Actual inference backend libraries (libonnxruntime.so, libnvinfer.so, etc.)
# are linked by the InferenceEngines library, not this project.
# This project only includes the backend-specific source directories.

