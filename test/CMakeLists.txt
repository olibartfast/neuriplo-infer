enable_testing()

include(GoogleTest)

set(PROJECT_NAME_TEST runUnitTests)

# Add test executable
add_executable(${PROJECT_NAME_TEST}
    ${CMAKE_CURRENT_LIST_DIR}/test_DetectorSetup.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/test_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_parseCommandLineArguments.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_ObjectDetectionApp.cpp    
    ${CMAKE_SOURCE_DIR}/src/utils.cpp 
    ${CMAKE_SOURCE_DIR}/src/ObjectDetectionApp.cpp
    ${CMAKE_SOURCE_DIR}/src/DetectorSetup.cpp
)

target_include_directories(${PROJECT_NAME_TEST} 
    PUBLIC ${CMAKE_SOURCE_DIR}/inc 
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_SOURCE_DIR}/detectors/src
    ${VideoCapture_SOURCE_DIR}/include
    ${VideoCapture_SOURCE_DIR}/src
    ${InferenceEngines_SOURCE_DIR}/backends/src
    ${InferenceEngines_SOURCE_DIR}/backends
    ${InferenceEngines_SOURCE_DIR}/include
    ${InferenceEngines_SOURCE_DIR}/src)

# Include framework-specific source files and libraries
if (DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
    target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${InferenceEngines_SOURCE_DIR}/backends/opencv-dnn/src)
elseif (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${ONNX_RUNTIME_DIR}/include ${InferenceEngines_SOURCE_DIR}/backends/onnx-runtime/src)
    target_link_directories(${PROJECT_NAME_TEST} PRIVATE ${ONNX_RUNTIME_DIR}/lib)
    target_link_libraries(${PROJECT_NAME_TEST} PRIVATE ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so)
elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${InferenceEngines_SOURCE_DIR}/backends/libtorch/src)
    target_link_libraries(${PROJECT_NAME_TEST} PRIVATE ${TORCH_LIBRARIES})
elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    target_include_directories(${PROJECT_NAME_TEST} PRIVATE /usr/local/cuda/include ${TENSORRT_DIR}/include ${InferenceEngines_SOURCE_DIR}/backends/tensorrt/src)
    target_link_directories(${PROJECT_NAME_TEST} PRIVATE  /usr/local/cuda/lib64 ${TENSORRT_DIR}/lib)
    target_link_libraries(${PROJECT_NAME_TEST} PRIVATE nvinfer nvonnxparser cudart)
elseif(DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW" )
    target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${TensorFlow_INCLUDE_DIRS} ${InferenceEngines_SOURCE_DIR}/backends/libtensorflow/src)
    target_link_libraries(${PROJECT_NAME_TEST} PRIVATE ${TensorFlow_LIBRARIES})  
elseif(DEFAULT_BACKEND STREQUAL "OPENVINO")
    target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${InferenceEngine_INCLUDE_DIRS} ${InferenceEngines_SOURCE_DIR}/backends/openvino/src)
    target_link_libraries(${PROJECT_NAME_TEST} PRIVATE openvino::runtime )
endif()

target_link_libraries(${PROJECT_NAME_TEST} PUBLIC gtest gtest_main glog::glog ${OpenCV_LIBS} DetectorsLib VideoCapture InferenceEngines  )


# Discover and run tests
gtest_discover_tests(${PROJECT_NAME_TEST})