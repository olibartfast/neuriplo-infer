# Add source files for detectors
set(DETECTORS_SOURCES 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/YoloNas.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/RtDetr.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/RtDetrUltralytics.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/RfDetr.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/YoloV4.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/src/models/YoloVn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/models/YOLOv10.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/DetectorSetup.cpp 
)

# Create the detectors library
add_library(detectors SHARED ${DETECTORS_SOURCES})


message(STATUS "Detectors Module, CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "Detectors Module, InferenceEngines_SOURCE_DIR: ${InferenceEngines_SOURCE_DIR}")

target_include_directories(detectors PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${InferenceEngines_SOURCE_DIR}/backends/src
    ${CMAKE_CURRENT_LIST_DIR}/inc
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/src/models
)

target_link_libraries(detectors PRIVATE
    ${OpenCV_LIBS}
    glog::glog
)

