cmake_minimum_required(VERSION 3.0)
project (tensorrt_test)

add_definitions(-std=c++17)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

add_executable(${PROJECT_NAME} tensorrt_test.cpp)