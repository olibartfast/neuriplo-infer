# List of supported backends
set(SUPPORTED_BACKENDS "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "OPENCV_DNN" "TENSORRT")

# Include backend-specific test CMakeLists.txt files
foreach(backend ${SUPPORTED_BACKENDS})
    if(${backend} STREQUAL ${DEFAULT_BACKEND})
        include(${backend}_Test.cmake)
    endif()
endforeach()