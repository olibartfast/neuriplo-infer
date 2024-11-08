# Add source files for detectors
set(DETECTORS_SOURCES 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/YoloNas.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/RtDetr.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/RtDetrUltralytics.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/YoloV4.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/YoloVn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/models/YOLOv10.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/DetectorSetup.cpp 
)

# Create the detectors library
add_library(detectors SHARED ${DETECTORS_SOURCES})

target_include_directories(detectors PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/inc
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/src/models
)

target_link_libraries(detectors PRIVATE
    ${OpenCV_LIBS}
    glog::glog
)

