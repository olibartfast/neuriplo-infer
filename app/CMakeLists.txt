# Fetch the VideoCapture project from GitHub with specific version
include(FetchContent)
FetchContent_Declare(
    VideoCapture
    GIT_REPOSITORY https://github.com/olibartfast/VideoCapture.git
    GIT_TAG        ${VIDEOCAPTURE_VERSION}  
)
# Pass video backend options to VideoCapture library
set(USE_GSTREAMER ${USE_GSTREAMER})
set(USE_FFMPEG ${USE_FFMPEG})
FetchContent_MakeAvailable(VideoCapture)
message(STATUS "VideoCapture_SOURCE_DIR: ${VideoCapture_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR:  ${CMAKE_CURRENT_SOURCE_DIR}")

# Define sources using CMAKE_CURRENT_SOURCE_DIR instead of CMAKE_SOURCE_DIR
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VisionApp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/InferencePipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CLICommands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ResultRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VisionAppTaskRouting.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CommandLineParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KserveClient.cpp
)

if(VISION_INFERENCE_WITH_GRPC)
    set(PROTO_FILE "${CMAKE_SOURCE_DIR}/proto/kserve_grpc.proto")
    get_filename_component(PROTO_DIR "${PROTO_FILE}" DIRECTORY)
    set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto")
    file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

    set(PROTO_SRCS "${PROTO_GEN_DIR}/kserve_grpc.pb.cc")
    set(GRPC_SRCS "${PROTO_GEN_DIR}/kserve_grpc.grpc.pb.cc")

    get_target_property(_GRPC_PLUGIN_SUPPORT_LOC gRPC::grpc_plugin_support IMPORTED_LOCATION_NONE)
    if(NOT _GRPC_PLUGIN_SUPPORT_LOC)
        get_target_property(_GRPC_PLUGIN_SUPPORT_LOC gRPC::grpc_plugin_support IMPORTED_LOCATION)
    endif()
    if(_GRPC_PLUGIN_SUPPORT_LOC)
        get_filename_component(_GRPC_LIB_DIR "${_GRPC_PLUGIN_SUPPORT_LOC}" DIRECTORY)
    else()
        set(_GRPC_LIB_DIR "")
    endif()

    if(_GRPC_LIB_DIR)
        add_custom_command(
            OUTPUT "${PROTO_SRCS}" "${GRPC_SRCS}"
            COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${_GRPC_LIB_DIR}"
                "${PROTOBUF_PROTOC_EXECUTABLE}"
                "--proto_path=${PROTO_DIR}"
                "--cpp_out=${PROTO_GEN_DIR}"
                "--grpc_out=${PROTO_GEN_DIR}"
                "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
                "${PROTO_FILE}"
            DEPENDS "${PROTO_FILE}"
            COMMENT "Generating gRPC sources"
        )
    else()
        add_custom_command(
            OUTPUT "${PROTO_SRCS}" "${GRPC_SRCS}"
            COMMAND "${PROTOBUF_PROTOC_EXECUTABLE}"
                "--proto_path=${PROTO_DIR}"
                "--cpp_out=${PROTO_GEN_DIR}"
                "--grpc_out=${PROTO_GEN_DIR}"
                "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
                "${PROTO_FILE}"
            DEPENDS "${PROTO_FILE}"
            COMMENT "Generating gRPC sources"
        )
    endif()

    list(APPEND SOURCES "${PROTO_SRCS}" "${GRPC_SRCS}"
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KserveGrpcClient.cpp)
endif()
add_executable(${PROJECT_NAME} ${SOURCES})
include(${CMAKE_SOURCE_DIR}/cmake/Sanitizers.cmake)
enable_target_sanitizers(${PROJECT_NAME})


# Set include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${VideoCapture_SOURCE_DIR}/include
    ${VideoCapture_SOURCE_DIR}/src
    ${neuriplo_SOURCE_DIR}/backends
    ${neuriplo_SOURCE_DIR}/backends/src
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    glog::glog
    ${OpenCV_LIBS}
    neuriplo
    VideoCapture
    vision-core
    nlohmann_json::nlohmann_json
)

if(VISION_INFERENCE_WITH_GRPC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE VISION_INFERENCE_WITH_GRPC)
    target_include_directories(${PROJECT_NAME} PRIVATE "${PROTO_GEN_DIR}")
    target_link_directories(${PROJECT_NAME} PRIVATE "/tmp/grpc-sysroot/usr/lib/x86_64-linux-gnu")
    target_link_libraries(${PROJECT_NAME} PRIVATE
        protobuf::libprotobuf
        gRPC::grpc++
    )
endif()

include(${CMAKE_SOURCE_DIR}/cmake/LinkBackend.cmake)

if(ENABLE_APP_TESTS)
    add_subdirectory(test)
endif()
