cmake_minimum_required(VERSION 3.0)
project (object-detection-inference)

find_package( OpenCV REQUIRED )
find_package(PkgConfig REQUIRED)
pkg_search_module(GSTREAMER gstreamer-1.0)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_search_module(GLIB REQUIRED glib-2.0)
pkg_search_module(GOBJECT REQUIRED gobject-2.0)
set(CMAKE_CXX_STANDARD 17)

set(SOURCES "main.cpp" "src/GStreamerOpenCV.cpp" "src/Detector.cpp"  "src/YoloV4.cpp" "src/YoloV5.cpp" "src/HogSvmDetector.cpp" "src/MobileNetSSD.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(
    ${PROJECT_NAME} PRIVATE 
    inc 
    ${OpenCV_INCLUDE_DIRS} 
    ${GSTREAMER_INCLUDE_DIRS} 
    ${GST_APP_INCLUDE_DIRS} 
    ${GST_VIDEO_INCLUDE_DIRS} 
    ${GLIB_INCLUDE_DIRS} ${GOBJECT_INCLUDE_DIR})

add_definitions(${GLIB_CFLAGS_OTHER})

target_link_libraries(
    ${PROJECT_NAME} ${OpenCV_LIBS} 
    ${GSTREAMER_LIBRARIES}  
    ${GST_APP_LIBRARIES} 
    ${GST_VIDEO_LIBRARIES} 
    ${GLIB_LIBRARIES}  ${GOBJECT_LIBRARIES}
)
