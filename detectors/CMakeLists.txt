# Add source files for detectors
set(DETECTORS_SOURCES 
    src/models/YoloNas.cpp 
    src/models/RtDetr.cpp 
    src/models/RtDetrUltralytics.cpp 
    src/models/YoloV4.cpp 
    src/models/YoloVn.cpp
    src/models/YOLOv10.cpp
    src/DetectorSetup.cpp 
)

# Create the detectors library
add_library(detectors SHARED ${DETECTORS_SOURCES})

target_include_directories(detectors PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/detectors/inc
    ${CMAKE_SOURCE_DIR}/detectors/src
    ${CMAKE_SOURCE_DIR}/detectors/src/models
)

target_link_libraries(detectors PRIVATE
    ${OpenCV_LIBS}
    glog::glog
)

