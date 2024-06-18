
# Add source files for video capture
set(VIDEOCAPTURE_SOURCES VideoCaptureInterface.cpp )

# Create the video capture library
add_library(VideoCaptureLib SHARED ${VIDEOCAPTURE_SOURCES})

target_include_directories(VideoCaptureLib PRIVATE
    ${CMAKE_SOURCE_DIR}/inc
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}      
    src/videocapture
)

# Link against GStreamer libraries if USE_GSTREAMER is ON
if (USE_GSTREAMER)
    target_include_directories(VideoCaptureLib PRIVATE
        ${GSTREAMER_INCLUDE_DIRS}
        ${GST_APP_INCLUDE_DIRS}
        ${GST_VIDEO_INCLUDE_DIRS}
    )
    target_link_libraries(VideoCaptureLib PRIVATE
        ${GSTREAMER_LIBRARIES}
        ${GST_APP_LIBRARIES}
        ${GST_VIDEO_LIBRARIES}
    )
endif()

target_link_libraries(DetectorsLib PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
)
