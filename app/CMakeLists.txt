set(APP_LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/NeuriploInfer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/InferencePipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CLICommands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ResultRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/NeuriploInferTaskRouting.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CommandLineParser.cpp
)

set(NEURIPLO_INFER_LIBRARY neuriplo-infer)
set(NEURIPLO_INFER_EXECUTABLE neuriplo-infer-cli)

add_library(${NEURIPLO_INFER_LIBRARY} STATIC ${APP_LIB_SOURCES})

target_include_directories(${NEURIPLO_INFER_LIBRARY} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${VideoCapture_SOURCE_DIR}/include
    ${VideoCapture_SOURCE_DIR}/src
    ${neuriplo_SOURCE_DIR}/backends
    ${neuriplo_SOURCE_DIR}/backends/src
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(${NEURIPLO_INFER_LIBRARY} PUBLIC
    glog::glog
    ${OpenCV_LIBS}
    neuriplo
    VideoCapture
    neuriplo-tasks
)

if(MSVC)
    target_compile_options(${NEURIPLO_INFER_LIBRARY} PRIVATE /W4)
else()
    target_compile_options(${NEURIPLO_INFER_LIBRARY} PRIVATE ${PROJECT_WARNING_FLAGS})
endif()

add_executable(${NEURIPLO_INFER_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set_target_properties(${NEURIPLO_INFER_EXECUTABLE} PROPERTIES OUTPUT_NAME neuriplo-infer)

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizers.cmake)
enable_target_sanitizers(${NEURIPLO_INFER_EXECUTABLE})

target_include_directories(${NEURIPLO_INFER_EXECUTABLE} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${NEURIPLO_INFER_LIBRARY})

if(MSVC)
    target_compile_options(${NEURIPLO_INFER_EXECUTABLE} PRIVATE /W4)
else()
    target_compile_options(${NEURIPLO_INFER_EXECUTABLE} PRIVATE ${PROJECT_WARNING_FLAGS})
endif()

include(${CMAKE_SOURCE_DIR}/cmake/LinkBackend.cmake)

if(ENABLE_APP_TESTS)
    add_subdirectory(test)
endif()
