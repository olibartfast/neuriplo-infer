# LibTorch Configuration

# Set LibTorch directory (modify accordingly)
set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch/)

# Find LibTorch
find_package(Torch REQUIRED)


set(LIBTORCH_SOURCES
    src/libtorch/LibtorchInfer.cpp
    # Add more LibTorch source files here if needed
)

# Append ONNX Runtime sources to the main sources
list(APPEND SOURCES ${LIBTORCH_SOURCES})


# Add compile definition to indicate LibTorch usage
add_compile_definitions(USE_LIBTORCH)
