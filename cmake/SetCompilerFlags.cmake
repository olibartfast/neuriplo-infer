# Set the appropriate compiler flags
if (CMAKE_CUDA_COMPILER AND DEFAULT_BACKEND STREQUAL "TENSORRT")
    # If CUDA is available and TensorRT is selected, set the CUDA flags
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
else()
    # If CUDA is not available or a different framework is selected, set the CPU flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()