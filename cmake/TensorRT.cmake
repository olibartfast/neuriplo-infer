# TensorRT Configuration

# Set TensorRT directory (modify accordingly)
set(TENSORRT_DIR $ENV{HOME}/TensorRT-8.6.1.6/)

# Find CUDA
find_package(CUDA REQUIRED)
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_COMPUTE_CAP
    RESULT_VARIABLE GPU_COMPUTE_CAP_RESULT
)

if (GPU_COMPUTE_CAP_RESULT EQUAL 0)
    # Split the GPU compute capabilities into a list
    string(REPLACE "\n" ";" GPU_COMPUTE_CAP_LIST ${GPU_COMPUTE_CAP})

    foreach(GPU_CAP ${GPU_COMPUTE_CAP_LIST})
        string(STRIP ${GPU_CAP} GPU_CAP)  # Remove leading/trailing whitespace
        message("GPU Compute Capability: ${GPU_CAP}")

        # Extract the major and minor compute capability values
        string(REGEX REPLACE "\\." ";" COMP_CAP_LIST ${GPU_CAP})
        list(GET COMP_CAP_LIST 0 COMPUTE_CAP_MAJOR)
        list(GET COMP_CAP_LIST 1 COMPUTE_CAP_MINOR)

        # Set CUDA flags based on the detected compute capability for each GPU
        set(CUDA_COMPUTE "compute_${COMPUTE_CAP_MAJOR}${COMPUTE_CAP_MINOR}")
        set(CUDA_SM "sm_${COMPUTE_CAP_MAJOR}${COMPUTE_CAP_MINOR}")
        message("Setting -gencode;arch=${CUDA_COMPUTE};code=${CUDA_SM} for GPU ${GPU_CAP}")

        # You can set CUDA flags differently for each GPU or collect them in a list or dictionary.
        # Here, we print the CUDA flags for each GPU.

        # Set CUDA flags for release for each GPU
        set(CUDA_NVCC_FLAGS_RELEASE_${GPU_CAP} ${CUDA_NVCC_FLAGS_RELEASE};-O3;-gencode;arch=${CUDA_COMPUTE};code=${CUDA_SM})

        # Set CUDA flags for debug for each GPU
        set(CUDA_NVCC_FLAGS_DEBUG_${GPU_CAP} ${CUDA_NVCC_FLAGS_DEBUG};-g;-G;-gencode;arch=${CUDA_COMPUTE};code=${CUDA_SM})
    endforeach()
else()
    message("Failed to query GPU compute capability.")
endif()    




set(TENSORRT_SOURCES
    src/tensorrt/TRTInfer.cpp
    # Add more TensorRT source files here if needed
)
list(APPEND SOURCES ${TENSORRT_SOURCES})

# Add compile definition to indicate TensorRT usage
add_compile_definitions(USE_TENSORRT)
