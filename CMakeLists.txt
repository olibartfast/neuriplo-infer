cmake_minimum_required(VERSION 3.20)
project(object-detection-inference)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# Define an option to enable or disable GStreamer support (default is ON)
unset(USE_GSTREAMER CACHE)
option(USE_GSTREAMER "Use GStreamer for video capture (optional)" ON)

# Include GStreamer-related settings and source files if USE_GSTREAMER is ON
if (USE_GSTREAMER)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(GSTREAMER gstreamer-1.0)
    pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
    pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
    pkg_search_module(GLIB REQUIRED glib-2.0)
    pkg_search_module(GOBJECT REQUIRED gobject-2.0)

    # Define GStreamer-related source files
    set(GST_SOURCE_FILES
        #src/GStreamerCapture.cpp
        src/GStreamerOpenCV.cpp
        # Add more GStreamer source files here if needed
    )

    # Set the appropriate sources, including GStreamer if enabled
    set(SOURCES main.cpp ${GST_SOURCE_FILES} src/Detector.cpp src/Yolo.cpp)

    # Define a compile definition to indicate GStreamer usage
    add_compile_definitions(USE_GSTREAMER)
else()
    # Set the appropriate sources without GStreamer
    set(SOURCES main.cpp src/Detector.cpp src/Yolo.cpp)
endif()

# Set the path to the selected framework (modify accordingly)
set(DEFAULT_BACKEND "ONNX_RUNTIME")  # Options: ONNX_RUNTIME, LIBTORCH, TENSORRT, LIBTENSORFLOW, OPENCV_DNN

# Define the supported backends
set(SUPPORTED_BACKENDS "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "OPENCV_DNN" "TENSORRT")

# Check if the specified backend is supported
list(FIND SUPPORTED_BACKENDS ${DEFAULT_BACKEND} SUPPORTED_BACKEND_INDEX)
if (SUPPORTED_BACKEND_INDEX EQUAL -1)
    message(STATUS "Unsupported default backend: ${DEFAULT_BACKEND}")
    set(DEFAULT_BACKEND "OPENCV_DNN")
endif()

# Unset cache compiler definitions for the selected framework
if (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    unset(USE_LIBTORCH CACHE)
    unset(USE_TENSORRT CACHE)
    unset(USE_OPENCV_DNN CACHE)
    unset(USE_TENSORFLOW CACHE)
    message(STATUS "Set ONNX Runtime")
    add_compile_definitions(USE_ONNX_RUNTIME)
    # Set the path to ONNX Runtime (modify accordingly)
    set(ONNX_RUNTIME_DIR $ENV{HOME}/onnxruntime-linux-x64-gpu-1.15.1)
    find_package(CUDA)
    if (CUDA_FOUND)
        message(STATUS "Found CUDA")
        set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
    else ()
        message(WARNING "CUDA not found. GPU support will be disabled.")
    endif ()
    include_directories(${ONNX_RUNTIME_DIR}/include src/onnx-runtime)
    link_directories(${ONNX_RUNTIME_DIR}/lib)

elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    unset(USE_ONNX_RUNTIME CACHE)
    unset(USE_TENSORRT CACHE)
    unset(USE_OPENCV_DNN CACHE)
    unset(USE_TENSORFLOW CACHE)
    message(STATUS "Set libtorch")
    add_compile_definitions(USE_LIBTORCH)
    # Set the path to LibTorch (modify accordingly)
    set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch/)
    message(STATUS "${Torch_DIR}")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    include_directories(src/libtorch)

elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    unset(USE_ONNX_RUNTIME CACHE)
    unset(USE_LIBTORCH CACHE)
    unset(USE_OPENCV_DNN CACHE)
    unset(USE_TENSORFLOW CACHE)
    message(STATUS "Set tensorrt")
    add_compile_definitions(USE_TENSORRT)
    # Set the path to TensorRT (modify accordingly)
    set(TENSORRT_DIR $ENV{HOME}/TensorRT-8.6.1.6/)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-g;-G;-gencode;arch=compute_75;code=sm_75)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    include_directories(${TENSORRT_DIR}/include src/tensorrt)
    link_directories(${TENSORRT_DIR}/lib)

elseif (DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW")
    unset(USE_OPENCV_DNN CACHE)
    unset(USE_LIBTORCH CACHE)
    unset(USE_TENSORRT CACHE)
    unset(USE_ONNX_RUNTIME CACHE)
    add_compile_definitions(USE_TENSORFLOW)
    find_package(TensorFlow REQUIRED)
    message(STATUS ${TensorFlow_FOUND})
    include_directories(src/libtensorflow)
endif()

add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    inc
    src
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

# Link against GStreamer libraries if USE_GSTREAMER is ON
if (USE_GSTREAMER)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${GSTREAMER_INCLUDE_DIRS}
        ${GST_APP_INCLUDE_DIRS}
        ${GST_VIDEO_INCLUDE_DIRS}
    )
    target_link_libraries(${PROJECT_NAME}
        ${GSTREAMER_LIBRARIES}
        ${GST_APP_LIBRARIES}
        ${GST_VIDEO_LIBRARIES}
    )
endif()

# Include framework-specific source files and libraries
if (DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
    set(OPENCV_DNN_SOURCES
        src/opencv-dnn/YoloVn.cpp
        src/opencv-dnn/YoloNas.cpp
        src/opencv-dnn/YoloV8.cpp
        src/opencv-dnn/YoloV4.cpp
        # Add more OpenCV DNN source files here if needed
    )
    target_sources(${PROJECT_NAME} PRIVATE ${OPENCV_DNN_SOURCES})
    include_directories(src/opencv-dnn)

elseif (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    set(ONNX_RUNTIME_SOURCES
        src/onnx-runtime/YoloV8.cpp
        src/onnx-runtime/YoloNas.cpp
        src/onnx-runtime/RtDetr.cpp
        # Add more ONNX Runtime source files here if needed
    )
    target_sources(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_SOURCES})
    target_link_libraries(${PROJECT_NAME}
        ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so
    )

elseif (DEFAULT_BACKEND STREQUAL "LIBTORCH")
    set(LIBTORCH_SOURCES
        src/libtorch/YoloV8.cpp
        src/libtorch/RtDetr.cpp
        src/libtorch/YoloVn.cpp
        # Add more LibTorch source files here if needed
    )
    target_sources(${PROJECT_NAME} PRIVATE ${LIBTORCH_SOURCES})
    target_link_libraries(${PROJECT_NAME}
        ${TORCH_LIBRARIES}
    )

elseif (DEFAULT_BACKEND STREQUAL "TENSORRT")
    set(TENSORRT_SOURCES
        src/tensorrt/YoloV8.cpp
        src/tensorrt/RtDetr.cpp
        # Add more TensorRT source files here if needed
    )
    target_sources(${PROJECT_NAME} PRIVATE ${TENSORRT_SOURCES})
    target_link_libraries(${PROJECT_NAME}
        nvinfer
        nvonnxparser
        cudart
    )
endif()

# Set the appropriate compiler flags
if (CMAKE_CUDA_COMPILER AND DEFAULT_BACKEND STREQUAL "TENSORRT")
    # If CUDA is available and TensorRT is selected, set the CUDA flags
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
else()
    # If CUDA is not available or a different framework is selected, set the CPU flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Add the TensorFlow sources to our target if the option is enabled
if (TensorFlow_FOUND AND DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW")
    set(TensorFlow_SOURCES
        src/libtensorflow/TFDetectionAPI.cpp
        # Add more TensorFlow source files here if needed
    )
    target_sources(${PROJECT_NAME} PRIVATE ${TensorFlow_SOURCES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${TensorFlow_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${TensorFlow_LIBRARIES})
endif()
