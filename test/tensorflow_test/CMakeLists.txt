cmake_minimum_required(VERSION 3.0)

project(tensorflow_test)
find_package(TensorFlow REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/test.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${TensorFlow_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TensorFlow_LIBRARIES}  ${OpenCV_LIBS})