# Fetch the VideoCapture project from GitHub with specific version
include(FetchContent)
FetchContent_Declare(
    VideoCapture
    GIT_REPOSITORY https://github.com/olibartfast/VideoCapture.git
    GIT_TAG        ${VIDEOCAPTURE_VERSION}  
)
set(USE_GSTREAMER ${USE_GSTREAMER})
FetchContent_MakeAvailable(VideoCapture)
message(STATUS "VideoCapture_SOURCE_DIR: ${VideoCapture_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR:  ${CMAKE_CURRENT_SOURCE_DIR}")

# Define sources using CMAKE_CURRENT_SOURCE_DIR instead of CMAKE_SOURCE_DIR
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ObjectDetectionApp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CommandLineParser.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})


# Set include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${VideoCapture_SOURCE_DIR}/include
    ${VideoCapture_SOURCE_DIR}/src
    ${InferenceEngines_SOURCE_DIR}/backends
    ${InferenceEngines_SOURCE_DIR}/backends/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../detectors/inc  # Relative path to detectors/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../detectors/src/models  # Relative path to detectors models
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${GLOG_LIBRARIES}
    ${OpenCV_LIBS}
    detectors
    InferenceEngines
    VideoCapture
)

# Option to enable unit tests for the app module
option(ENABLE_APP_TESTS "Enable unit testing for app module" OFF)
unset(ENABLE_APP_TESTS CACHE) # Clear any previous cache

if(ENABLE_APP_TESTS)
    enable_testing()
    add_subdirectory(app/test)
endif()
