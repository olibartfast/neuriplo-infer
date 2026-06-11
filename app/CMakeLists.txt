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
    # The KServe protocol client (HTTP/gRPC transports, proto, TLS) lives in the
    # standalone neuriplo-kserve-client library, fetched in the top-level
    # CMakeLists. Only the adapter that bridges it to the neuriplo inference
    # contract stays in this app.
    list(APPEND APP_LIB_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KserveEngine.cpp)
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
    # The KServe client library carries the HTTP/gRPC transports and its public
    # include dir + KSERVE_CLIENT_WITH_GRPC define propagate to KserveEngine and
    # InferencePipeline.
    target_link_libraries(${NEURIPLO_INFER_LIBRARY} PUBLIC neuriplo-kserve-client)
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
