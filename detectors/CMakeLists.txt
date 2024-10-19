# Add source files for detectors
set(DETECTORS_SOURCES 
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
    ${GLOG_INCLUDE_DIRS}
    detectors
)

target_link_libraries(DetectorsLib PRIVATE
    ${OpenCV_LIBS}
    glog::glog
)

# Option to enable unit tests
option(ENABLE_DETECTORS_TESTS "Enable unit testing for detectors lib" OFF)

if(ENABLE_DETECTORS_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()