cmake_minimum_required(VERSION 3.20)
project (object-detection-inference)

find_package( OpenCV REQUIRED )
find_package(PkgConfig REQUIRED)
pkg_search_module(GSTREAMER gstreamer-1.0)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_search_module(GLIB REQUIRED glib-2.0)
pkg_search_module(GOBJECT REQUIRED gobject-2.0)
set(CMAKE_CXX_STANDARD 17)

# Set the path to the selected framework (modify accordingly)
set(DEFAULT_BACKEND "ONNX_RUNTIME")  # Options: ONNX_RUNTIME, LIBTORCH, TENSORRT, LIBTENSORFLOW, OPENCV_DNN

# Define the supported backends
set(SUPPORTED_BACKENDS "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "OPENCV_DNN" "TENSORRT")

# Check if the specified backend is supported
list(FIND SUPPORTED_BACKENDS ${DEFAULT_BACKEND} SUPPORTED_BACKEND_INDEX)
if(SUPPORTED_BACKEND_INDEX EQUAL -1)
    message(STATUS "Unsupported default backend: ${DEFAULT_BACKEND}")
    set(DEFAULT_BACKEND "OPENCV_DNN") 
endif()

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS} include src)

# Unset cache compiler definitions for the selected framework
if (DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
    unset(USE_LIBTORCH CACHE)
    unset(USE_TENSORRT CACHE)
    unset(USE_OPENCV_DNN CACHE)
    unset(USE_TENSORFLOW CACHE)
    message(STATUS ": Set ONNX Runtime")
    add_compile_definitions(USE_ONNX_RUNTIME)
    # Set the path to ONNX Runtime (modify accordingly)
    set(ONNX_RUNTIME_DIR $ENV{HOME}/onnxruntime-linux-x64-gpu-1.15.1)
    find_package(CUDA)
    if (CUDA_FOUND)
        message(STATUS "Found CUDA")
        set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
    else ()
        message(WARNING "CUDA not found. GPU support will be disabled.")
        # Handle the case when CUDA is not found
        # You can set alternative behavior or show an error message
        # depending on your project requirements.
    endif ()
    include_directories(${ONNX_RUNTIME_DIR}/include src/onnx-runtime)
    link_directories(${ONNX_RUNTIME_DIR}/lib)

elseif(DEFAULT_BACKEND STREQUAL "LIBTORCH")
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

elseif(DEFAULT_BACKEND STREQUAL "TENSORRT")
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
elseif(DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW")
    unset(USE_OPENCV_DNN CACHE)
    unset(USE_LIBTORCH CACHE)
    unset(USE_TENSORRT CACHE)
    unset(USE_ONNX_RUNTIME CACHE)
    add_compile_definitions(USE_TENSORFLOW)
    find_package(TensorFlow REQUIRED)
    message(status ${TensorFlow_FOUND})
    include_directories(src/libtensorflow)
else()
    unset(USE_LIBTORCH CACHE)
    unset(USE_TENSORRT CACHE)
    unset(USE_ONNX_RUNTIME CACHE)
    unset(USE_TENSORFLOW CACHE)
    message(status ": using ${DEFAULT_BACKEND} as default inference backend")
    add_compile_definitions(USE_OPENCV_DNN)
endif()


set(SOURCES main.cpp src/GStreamerOpenCV.cpp src/Detector.cpp src/Yolo.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(
    ${PROJECT_NAME} PRIVATE 
    inc src
    ${OpenCV_INCLUDE_DIRS} 
    ${GSTREAMER_INCLUDE_DIRS} 
    ${GST_APP_INCLUDE_DIRS} 
    ${GST_VIDEO_INCLUDE_DIRS} 
    ${GLIB_INCLUDE_DIRS} ${GOBJECT_INCLUDE_DIR})

add_definitions(${GLIB_CFLAGS_OTHER})

target_link_libraries(
    ${PROJECT_NAME} ${OpenCV_LIBS} 
    ${GSTREAMER_LIBRARIES}  
    ${GST_APP_LIBRARIES} 
    ${GST_VIDEO_LIBRARIES} 
    ${GLIB_LIBRARIES}  ${GOBJECT_LIBRARIES}
)

if(DEFAULT_BACKEND STREQUAL "OPENCV_DNN")
set(OPENCV_DNN_SOURCES src/opencv-dnn/YoloVn.cpp src/opencv-dnn/YoloNas.cpp src/opencv-dnn/YoloV8.cpp src/opencv-dnn/YoloV4.cpp)
target_sources(${PROJECT_NAME} PRIVATE ${OPENCV_DNN_SOURCES})
include_directories(src/opencv-dnn)
endif()

# Link against the ONNX Runtime library if selected
if(DEFAULT_BACKEND STREQUAL "ONNX_RUNTIME")
set(ONNX_RUNTIME_SOURCES src/onnx-runtime/YoloV8.cpp src/onnx-runtime/YoloNas.cpp)
target_sources(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_SOURCES})
    target_link_libraries(${PROJECT_NAME}
        ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so
    )
endif()

# Link against the LibTorch library if selected
if(DEFAULT_BACKEND STREQUAL "LIBTORCH")
set(LIBTORCH_SOURCES src/libtorch/YoloV8.cpp)
target_sources(${PROJECT_NAME} PRIVATE ${LIBTORCH_SOURCES})
    target_link_libraries(${PROJECT_NAME}
        ${TORCH_LIBRARIES}
    )
endif()

# Link against the TensorRT library if selected
if(DEFAULT_BACKEND STREQUAL "TENSORRT")
    set(TENSORRT_SOURCES src/tensorrt/YoloV8.cpp)
    target_sources(${PROJECT_NAME} PRIVATE ${TENSORRT_SOURCES})
    target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser cudart)
endif()

# Set the appropriate compiler flags
if(CMAKE_CUDA_COMPILER AND DEFAULT_BACKEND STREQUAL "TENSORRT")
    # If CUDA is available and TensorRT is selected, set the CUDA flags
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
else()
    # If CUDA is not available or a different framework is selected, set the CPU flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Add the TensorFlow sources to our target if the option is enabled
if(TensorFlow_FOUND AND DEFAULT_BACKEND STREQUAL "LIBTENSORFLOW")
  set(TensorFlow_SOURCES src/libtensorflow/TFDetectionAPI.cpp)
  target_sources(${PROJECT_NAME} PRIVATE ${TensorFlow_SOURCES})
  target_include_directories(
    ${PROJECT_NAME} PRIVATE  ${TensorFlow_INCLUDE_DIRS}) 
  target_link_libraries(${PROJECT_NAME} ${TensorFlow_LIBRARIES})  
endif()
