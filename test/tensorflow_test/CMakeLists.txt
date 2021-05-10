cmake_minimum_required(VERSION 3.0)

message("Libtensorflow env path " $ENV{LIBTENSORFLOW_DIR})
project(tensorflow_test)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/test.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC $ENV{LIBTENSORFLOW_DIR}/include $ENV{PROTOBUF_INC})
target_link_directories(${PROJECT_NAME}  PUBLIC $ENV{LIBTENSORFLOW_DIR})
target_link_libraries(${PROJECT_NAME} libtensorflow_cc.so libtensorflow_framework.so)