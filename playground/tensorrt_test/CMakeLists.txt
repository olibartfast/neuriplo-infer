cmake_minimum_required(VERSION 3.10)
project (tensorrt_test)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

find_package(OpenCV)
find_package(CUDA REQUIRED)

# Get CUDA architecture and code
execute_process(
  COMMAND ${CUDA_NVCC_EXECUTABLE} --run -arch
  OUTPUT_VARIABLE CUDA_ARCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${CUDA_NVCC_EXECUTABLE} --run -code
  OUTPUT_VARIABLE CUDA_CODE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set CUDA_NVCC_FLAGS
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-G;-gencode;arch=${CUDA_ARCH};code=${CUDA_CODE}")

message(STATUS "Set TensorRT Dir to:" $ENV{TENSORRT_DIR})

# EXECUTABLE
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/tensorrt_test.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC /usr/local/cuda/include ${OpenCV_INCLUDE_DIRS} $ENV{TENSORRT_DIR}/include)
target_link_directories(${PROJECT_NAME} PUBLIC /usr/local/cuda/lib64 $ENV{TENSORRT_DIR}/lib)
target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser cudart ${OpenCV_LIBS})