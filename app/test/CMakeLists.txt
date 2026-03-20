include(GoogleTest)
include(FetchContent)

find_package(GTest QUIET)
if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

if(TARGET gtest AND NOT TARGET GTest::gtest)
    add_library(GTest::gtest ALIAS gtest)
endif()
if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS gtest_main)
endif()

set(PROJECT_NAME_TEST runUnitTests)

# Add test executable
add_executable(${PROJECT_NAME_TEST}
    ${CMAKE_CURRENT_LIST_DIR}/test_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_parseCommandLineArguments.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_VisionApp.cpp    
    ${CMAKE_SOURCE_DIR}/app/src/utils.cpp 
    ${CMAKE_SOURCE_DIR}/app/src/CommandLineParser.cpp
    ${CMAKE_SOURCE_DIR}/app/src/VisionApp.cpp
    ${CMAKE_SOURCE_DIR}/app/src/VisionAppProcessing.cpp
    ${CMAKE_SOURCE_DIR}/app/src/VisionAppRendering.cpp
    ${CMAKE_SOURCE_DIR}/app/src/VisionAppTaskRouting.cpp
)


target_include_directories(${PROJECT_NAME_TEST} 
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/../inc 
    ${CMAKE_CURRENT_LIST_DIR}/../src 
    ${vision-core_SOURCE_DIR}/include
    ${VideoCapture_SOURCE_DIR}/include
    ${VideoCapture_SOURCE_DIR}/src
    ${neuriplo_SOURCE_DIR}/backends/src
    ${neuriplo_SOURCE_DIR}/backends
    ${neuriplo_SOURCE_DIR}/include
    ${neuriplo_SOURCE_DIR}/src)

# Include framework-specific source files and libraries
if (DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
    target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${neuriplo_SOURCE_DIR}/backends/opencv-dnn/src)
elseif (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    if(WIN32)
        message(WARNING "Skipping ONNX Runtime test linkage on Windows; use OPENCV_DNN for native Windows test builds.")
    else()
        target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${ONNX_RUNTIME_DIR}/include ${neuriplo_SOURCE_DIR}/backends/onnx-runtime/src)
        target_link_directories(${PROJECT_NAME_TEST} PRIVATE ${ONNX_RUNTIME_DIR}/lib)
        target_link_libraries(${PROJECT_NAME_TEST} PRIVATE ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so)
    endif()
elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${neuriplo_SOURCE_DIR}/backends/libtorch/src)
    target_link_libraries(${PROJECT_NAME_TEST} PRIVATE ${TORCH_LIBRARIES})
elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    if(WIN32)
        message(WARNING "Skipping TensorRT test linkage on Windows; use OPENCV_DNN for native Windows test builds.")
    else()
        target_include_directories(${PROJECT_NAME_TEST} PRIVATE /usr/local/cuda/include ${TENSORRT_DIR}/include ${neuriplo_SOURCE_DIR}/backends/tensorrt/src)
        target_link_directories(${PROJECT_NAME_TEST} PRIVATE  /usr/local/cuda/lib64 ${TENSORRT_DIR}/lib)
        target_link_libraries(${PROJECT_NAME_TEST} PRIVATE nvinfer nvonnxparser cudart)
    endif()
elseif(DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW" )
    if(WIN32)
        message(WARNING "Skipping TensorFlow test linkage on Windows; use OPENCV_DNN for native Windows test builds.")
    else()
        target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${TensorFlow_INCLUDE_DIRS} ${neuriplo_SOURCE_DIR}/backends/libtensorflow/src)
        target_link_libraries(${PROJECT_NAME_TEST} PRIVATE ${TensorFlow_LIBRARIES})  
    endif()
elseif(DEFAULT_BACKEND STREQUAL "OPENVINO")
    if(WIN32)
        message(WARNING "Skipping OpenVINO test linkage on Windows; use OPENCV_DNN for native Windows test builds.")
    else()
        target_include_directories(${PROJECT_NAME_TEST} PRIVATE ${InferenceEngine_INCLUDE_DIRS} ${neuriplo_SOURCE_DIR}/backends/openvino/src)
        target_link_libraries(${PROJECT_NAME_TEST} PRIVATE openvino::runtime )
    endif()
endif()

target_link_libraries(${PROJECT_NAME_TEST} PUBLIC GTest::gtest GTest::gtest_main glog::glog ${OpenCV_LIBS} vision-core VideoCapture neuriplo)

# Discover and run tests
gtest_discover_tests(${PROJECT_NAME_TEST})
