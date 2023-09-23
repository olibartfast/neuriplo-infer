# Include GStreamer-related settings and source files
find_package(PkgConfig REQUIRED)
pkg_search_module(GSTREAMER gstreamer-1.0)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_search_module(GLIB REQUIRED glib-2.0)
pkg_search_module(GOBJECT REQUIRED gobject-2.0)


# Define a compile definition to indicate GStreamer usage
add_compile_definitions(USE_GSTREAMER)

# Define GStreamer-specific source files
set(GST_SOURCE_FILES
    # src/GStreamerCapture.cpp
    src/GStreamerOpenCV.cpp
    # Add more GStreamer source files here if needed
)

# Append GStreamer sources to the main sources
list(APPEND SOURCES ${GST_SOURCE_FILES})