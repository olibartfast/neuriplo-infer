set(APP_LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/NeuriploInfer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/InferencePipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CLICommands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ResultRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/NeuriploInferTaskRouting.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CommandLineParser.cpp
)

if(NEURIPLO_INFER_ENABLE_KSERVE)
    list(APPEND APP_LIB_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KserveProtocol.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KserveRetry.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KserveHttpClient.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KserveEngine.cpp)
endif()

if(NEURIPLO_INFER_WITH_GRPC)
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

    list(APPEND APP_LIB_SOURCES
        "${PROTO_SRCS}"
        "${GRPC_SRCS}"
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KserveGrpcClient.cpp)
endif()

set(NEURIPLO_INFER_LIBRARY neuriplo-infer)
set(NEURIPLO_INFER_EXECUTABLE neuriplo-infer-cli)

add_library(${NEURIPLO_INFER_LIBRARY} STATIC ${APP_LIB_SOURCES})

target_include_directories(${NEURIPLO_INFER_LIBRARY} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${VideoCapture_SOURCE_DIR}/include
    ${VideoCapture_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

if(NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS)
    # neuriplo provides the inference contract and the local-engine factory.
    target_include_directories(${NEURIPLO_INFER_LIBRARY} PUBLIC
        ${neuriplo_SOURCE_DIR}/backends
        ${neuriplo_SOURCE_DIR}/backends/src
    )
else()
    # KServe-only: the contract comes from the app-local headers, nothing fetched.
    target_include_directories(${NEURIPLO_INFER_LIBRARY} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/contract
    )
endif()

target_link_libraries(${NEURIPLO_INFER_LIBRARY} PUBLIC
    glog::glog
    ${OpenCV_LIBS}
    VideoCapture
    neuriplo-tasks
    nlohmann_json::nlohmann_json
)

if(NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS)
    target_link_libraries(${NEURIPLO_INFER_LIBRARY} PUBLIC neuriplo)
    target_compile_definitions(${NEURIPLO_INFER_LIBRARY} PUBLIC
                               NEURIPLO_INFER_WITH_LOCAL_BACKENDS)
endif()

if(NEURIPLO_INFER_ENABLE_KSERVE)
    target_compile_definitions(${NEURIPLO_INFER_LIBRARY} PUBLIC NEURIPLO_INFER_WITH_KSERVE)
endif()

if(NEURIPLO_INFER_WITH_GRPC)
    target_compile_definitions(${NEURIPLO_INFER_LIBRARY} PRIVATE NEURIPLO_INFER_WITH_GRPC)
    target_include_directories(${NEURIPLO_INFER_LIBRARY} PRIVATE "${PROTO_GEN_DIR}")
    target_link_libraries(${NEURIPLO_INFER_LIBRARY} PRIVATE
        protobuf::libprotobuf
        gRPC::grpc++
    )
endif()

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
