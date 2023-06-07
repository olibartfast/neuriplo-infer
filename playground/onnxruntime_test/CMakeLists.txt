cmake_minimum_required(VERSION 3.0.0)
project(onnxruntime_test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "ONNXRUNTIME_DIR: $ENV{ONNXRUNTIME_DIR}")
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} onnxruntime_test.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE "$ENV{ONNXRUNTIME_DIR}/include")
target_link_directories(${PROJECT_NAME} PRIVATE "$ENV{ONNXRUNTIME_DIR}/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} onnxruntime)
