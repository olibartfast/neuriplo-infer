# Include framework-specific source files and libraries
# Note: Inference backend linking is handled by the neuriplo
# This file only includes backend-specific source directories from neuriplo

if (DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
    target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${neuriploDIR}/backends/opencv-dnn/src)
elseif (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${neuriplo_SOURCE_DIR}/backends/onnx-runtime/src)
elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${neuriplo_SOURCE_DIR}/backends/libtorch/src)
    target_compile_definitions(${NEURIPLO_INFER_EXECUTABLE}  PRIVATE C10_USE_GLOG)
elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${neuriploDIR}/backends/tensorrt/src)
elseif(DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW" )
    target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${neuriplo_SOURCE_DIR}/backends/libtensorflow/src)
elseif(DEFAULT_BACKEND STREQUAL "OPENVINO")
    target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${neuriplo_SOURCE_DIR}/backends/openvino/src)
elseif(DEFAULT_BACKEND STREQUAL "LITERT")
    target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${neuriplo_SOURCE_DIR}/backends/litert/src)
elseif(DEFAULT_BACKEND STREQUAL "LLAMACPP")
    target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${neuriplo_SOURCE_DIR}/backends/llamacpp/src)
    # libllama.so → libggml-base.so.0/libggml.so.0 are transitive SONAME deps; the final
    # executable must carry rpath-link so ld resolves them at link time, and rpath so the
    # dynamic loader finds them at run time.
    target_link_options(${NEURIPLO_INFER_EXECUTABLE} PRIVATE
        "-Wl,-rpath-link,${LLAMACPP_DIR}/lib"
        "-Wl,-rpath,${LLAMACPP_DIR}/lib")
    # Link ggml/llama/mtmd libs directly so undefined refs are satisfied.
    find_library(VI_GGML_LIB      NAMES ggml      PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
    find_library(VI_GGML_BASE_LIB NAMES ggml-base PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
    find_library(VI_GGML_CPU_LIB  NAMES ggml-cpu  PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
    find_library(VI_LLAMA_LIB     NAMES llama      PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
    find_library(VI_MTMD_LIB      NAMES mtmd       PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
    foreach(_lib VI_LLAMA_LIB VI_GGML_LIB VI_GGML_BASE_LIB VI_GGML_CPU_LIB VI_MTMD_LIB)
        if(${_lib})
            target_link_libraries(${NEURIPLO_INFER_EXECUTABLE} PRIVATE "${${_lib}}")
        endif()
    endforeach()
elseif(DEFAULT_BACKEND STREQUAL "EXECUTORCH")
    # ExecuTorch static libs are linked PRIVATE into libneuriplo.so by neuriplo's cmake,
    # so no additional link steps are needed here beyond the include path.
    target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${neuriplo_SOURCE_DIR}/backends/executorch/src)
endif()

# Note: Actual inference backend libraries (libonnxruntime.so, libnvinfer.so, etc.)
# are linked by the neuriplo library, not this project.
# This project only includes the backend-specific source directories.
