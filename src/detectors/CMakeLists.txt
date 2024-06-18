# Add source files for detectors
set(DETECTORS_SOURCES 
    Detector.cpp 
    YoloNas.cpp 
    RtDetr.cpp 
    RtDetrUltralytics.cpp 
    YoloV4.cpp 
    YoloVn.cpp
    YOLOv10.cpp
)

# Create the detectors library
add_library(DetectorsLib SHARED ${DETECTORS_SOURCES})

target_include_directories(DetectorsLib PRIVATE
    ${CMAKE_SOURCE_DIR}/inc
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    src/detectors
)

target_link_libraries(DetectorsLib PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
)
