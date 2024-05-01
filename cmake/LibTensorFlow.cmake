# TensorFlow Configuration

# Find TensorFlow
find_package(TensorFlow REQUIRED)


set(TensorFlow_SOURCES
src/inference-engines/libtensorflow/TFDetectionAPI.cpp
)

list(APPEND SOURCES ${TensorFlow_SOURCES})


# Add compile definition to indicate TensorFlow usage
add_compile_definitions(USE_LIBTENSORFLOW)

