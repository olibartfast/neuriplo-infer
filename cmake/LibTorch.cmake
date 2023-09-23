# LibTorch Configuration

# Set LibTorch directory (modify accordingly)
set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch/)

# Find LibTorch
find_package(Torch REQUIRED)

# Set C++ flags from LibTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(LIBTORCH_SOURCES
    src/libtorch/YoloV8.cpp
    src/libtorch/RtDetr.cpp
    src/libtorch/YoloVn.cpp
    # Add more LibTorch source files here if needed
)

# Append ONNX Runtime sources to the main sources
list(APPEND SOURCES ${LIBTORCH_SOURCES})


# Add compile definition to indicate LibTorch usage
add_compile_definitions(USE_LIBTORCH)
