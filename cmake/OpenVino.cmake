set(OPENVINO_SOURCES
src/openvino/OVInfer.cpp
# Add more OPENVINO source files here if needed
)

find_package(InferenceEngine REQUIRED)


list(APPEND SOURCES ${OPENVINO_SOURCES})

add_compile_definitions(USE_OPENVINO)