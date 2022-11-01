cmake_minimum_required (VERSION 3.10)
project(openvino_test)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${ov_link_libraries})