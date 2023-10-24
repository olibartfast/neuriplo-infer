add_executable(${PROJECT_NAME}-tests
    ONNXRuntimeTest.cpp
)

# Include directories for your tests
target_include_directories(${PROJECT_NAME}-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/inc  # Include your project's include directory
    ${CMAKE_SOURCE_DIR}/src/onnx-runtime
    ${OpenCV_INCLUDE_DIRS}
    ${ONNX_RUNTIME_DIR}/include 
    src/onnx-runtime
    ${spdlog_INCLUDE_DIRS}
)

target_link_directories(${PROJECT_NAME}-tests PRIVATE ${ONNX_RUNTIME_DIR}/lib)

# Link your tests with Google Test and any other necessary libraries
target_link_libraries(${PROJECT_NAME}-tests
    PRIVATE ${OpenCV_LIBS}  ${GTEST_BOTH_LIBRARIES}  ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so
    spdlog::spdlog_header_only
)


# Add your tests as a test target
add_test(NAME ${PROJECT_NAME}-tests COMMAND ${PROJECT_NAME}-tests)