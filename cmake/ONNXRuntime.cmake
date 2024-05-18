# ONNX Runtime Configuration
# Set ONNX Runtime
set(ORT_VERSION "1.15.1" CACHE STRING "Onnx runtime version") # modify accordingly
set(ONNX_RUNTIME_DIR $ENV{HOME}/onnxruntime-linux-x64-gpu-${ORT_VERSION} CACHE PATH "Path to onnxruntime")     
message(STATUS "Onnx runtime version: ${ORT_VERSION}")

# Set ONNX Runtime directory (modify accordingly)
message(STATUS "Onnx runtime directory: ${ONNX_RUNTIME_DIR}")


# Find CUDA (if available)
find_package(CUDA)
if (CUDA_FOUND)
    message(STATUS "Found CUDA")
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
else ()
    message(WARNING "CUDA not found. GPU support will be disabled.")
endif()

# Define ONNX Runtime-specific source files
set(ONNX_RUNTIME_SOURCES
    src/inference-engines/onnx-runtime/ORTInfer.cpp
    # Add more ONNX Runtime source files here if needed
)

# Append ONNX Runtime sources to the main sources
list(APPEND SOURCES ${ONNX_RUNTIME_SOURCES})

# Add compile definition to indicate ONNX Runtime usage
add_compile_definitions(USE_ONNX_RUNTIME)