if (CMAKE_CUDA_COMPILER)
    # If CUDA is available but not using TensorRT, set the CUDA flags
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Set any specific CUDA flags for non-TensorRT code here
    set(CUDA_ARCH_FLAG "--expt-extended-lambda") # CUDA compiler option that enables support for C++11 lambdas in device code.
else()
    # If CUDA is not available, set CPU flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CUDA_ARCH_FLAG "")
endif()

# Set common optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

# Set debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Combine CUDA flags with common flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAG}")

message("CMake CXX Flags: ${CMAKE_CXX_FLAGS}")
message("CMake CUDA Flags: ${CMAKE_CUDA_FLAGS}")
