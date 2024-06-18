
# Add source files for inference engines
set(INFERENCE_SOURCES InferenceInterface.cpp)

# Create the inference engines library
add_library(InferenceEnginesLib SHARED ${INFERENCE_SOURCES})

target_include_directories(InferenceEnginesLib PRIVATE
    ${CMAKE_SOURCE_DIR}/inc
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}    
    src/inference-engines
)

target_link_libraries(DetectorsLib PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
)
