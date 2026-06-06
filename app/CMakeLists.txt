set(APP_LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VisionApp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/InferencePipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CLICommands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ResultRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VisionAppTaskRouting.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CommandLineParser.cpp
)

add_library(vision-inference-app STATIC ${APP_LIB_SOURCES})

target_include_directories(vision-inference-app PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${VideoCapture_SOURCE_DIR}/include
    ${VideoCapture_SOURCE_DIR}/src
    ${neuriplo_SOURCE_DIR}/backends
    ${neuriplo_SOURCE_DIR}/backends/src
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(vision-inference-app PUBLIC
    glog::glog
    ${OpenCV_LIBS}
    neuriplo
    VideoCapture
    vision-core
)

if(MSVC)
    target_compile_options(vision-inference-app PRIVATE /W4)
else()
    target_compile_options(vision-inference-app PRIVATE ${PROJECT_WARNING_FLAGS})
endif()

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
include(${CMAKE_SOURCE_DIR}/cmake/Sanitizers.cmake)
enable_target_sanitizers(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE vision-inference-app)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE ${PROJECT_WARNING_FLAGS})
endif()

include(${CMAKE_SOURCE_DIR}/cmake/LinkBackend.cmake)

if(ENABLE_APP_TESTS)
    add_subdirectory(test)
endif()
