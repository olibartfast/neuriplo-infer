# Add source files for detectors
set(DETECTORS_SOURCES 
    src/Detector.cpp 
    src/YoloNas.cpp 
    src/RtDetr.cpp 
    src/RtDetrUltralytics.cpp 
    src/YoloV4.cpp 
    src/YoloVn.cpp
    src/YOLOv10.cpp
)

# Create the detectors library
add_library(DetectorsLib SHARED ${DETECTORS_SOURCES})

target_include_directories(DetectorsLib PRIVATE
    ${CMAKE_SOURCE_DIR}/inc
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    detectors
)

target_link_libraries(DetectorsLib PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
)
