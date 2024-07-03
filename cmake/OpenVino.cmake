set(OPENVINO_SOURCES
${INFER_ROOT}/src/openvino/OVInfer.cpp
# Add more OPENVINO source files here if needed
)

find_package(OpenVINO REQUIRED)


list(APPEND SOURCES ${OPENVINO_SOURCES})

add_compile_definitions(USE_OPENVINO)