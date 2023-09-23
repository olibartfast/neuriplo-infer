# ONNX Runtime Configuration

# Set ONNX Runtime directory (modify accordingly)
set(ONNX_RUNTIME_DIR $ENV{HOME}/onnxruntime-linux-x64-gpu-1.15.1)

# Find CUDA (if available)
find_package(CUDA)
if (CUDA_FOUND)
    message(STATUS "Found CUDA")
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
else ()
    message(WARNING "CUDA not found. GPU support will be disabled.")
endif()

# Define ONNX Runtime-specific source files
set(ONNX_RUNTIME_SOURCES
    src/onnx-runtime/YoloV8.cpp
    src/onnx-runtime/YoloNas.cpp
    src/onnx-runtime/RtDetr.cpp
    # Add more ONNX Runtime source files here if needed
)

# Append ONNX Runtime sources to the main sources
list(APPEND SOURCES ${ONNX_RUNTIME_SOURCES})

# Add compile definition to indicate ONNX Runtime usage
add_compile_definitions(USE_ONNX_RUNTIME)