# TensorRT Configuration

# Set TensorRT directory (modify accordingly)
set(TENSORRT_DIR $ENV{HOME}/TensorRT-8.6.1.6/)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G;-gencode;arch=compute_75;code=sm_75)

set(TENSORRT_SOURCES
    src/tensorrt/YoloV8.cpp
    src/tensorrt/RtDetr.cpp
    # Add more TensorRT source files here if needed
)
list(APPEND SOURCES ${TENSORRT_SOURCES})

# Add compile definition to indicate TensorRT usage
add_compile_definitions(USE_TENSORRT)
