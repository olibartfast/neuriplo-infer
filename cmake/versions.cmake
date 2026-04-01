# Centralized version management for this project's dependencies
# This file should be the single source of truth for versions managed by this project

function(read_versions_from_env)
    set(VERSIONS_ENV_FILE "${CMAKE_CURRENT_SOURCE_DIR}/versions.env")

    if(NOT EXISTS "${VERSIONS_ENV_FILE}")
        message(FATAL_ERROR "versions.env file not found at ${VERSIONS_ENV_FILE}")
    endif()

    file(READ "${VERSIONS_ENV_FILE}" VERSIONS_CONTENT)
    string(REPLACE "\n" ";" VERSIONS_LINES "${VERSIONS_CONTENT}")

    foreach(LINE ${VERSIONS_LINES})
        if(LINE AND NOT LINE MATCHES "^#")
            string(REGEX MATCH "^([A-Z_]+)=(.+)$" MATCH "${LINE}")
            if(MATCH)
                set(VAR_NAME "${CMAKE_MATCH_1}")
                set(VAR_VALUE "${CMAKE_MATCH_2}")
                string(REGEX REPLACE "^\"(.*)\"$" "\\1" VAR_VALUE "${VAR_VALUE}")
                set(${VAR_NAME} "${VAR_VALUE}" PARENT_SCOPE)
            endif()
        endif()
    endforeach()
endfunction()

function(get_current_branch_name OUTPUT_VAR)
    set(CURRENT_BRANCH "")

    if(DEFINED ENV{GITHUB_BASE_REF} AND NOT "$ENV{GITHUB_BASE_REF}" STREQUAL "")
        set(CURRENT_BRANCH "$ENV{GITHUB_BASE_REF}")
    elseif(DEFINED ENV{GITHUB_HEAD_REF} AND NOT "$ENV{GITHUB_HEAD_REF}" STREQUAL "")
        set(CURRENT_BRANCH "$ENV{GITHUB_HEAD_REF}")
    elseif(DEFINED ENV{GITHUB_REF_NAME} AND NOT "$ENV{GITHUB_REF_NAME}" STREQUAL "")
        set(CURRENT_BRANCH "$ENV{GITHUB_REF_NAME}")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/.git")
        execute_process(
            COMMAND git rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE CURRENT_BRANCH
            ERROR_VARIABLE GIT_ERROR
            RESULT_VARIABLE GIT_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(NOT GIT_RESULT EQUAL 0)
            string(STRIP "${GIT_ERROR}" GIT_ERROR)
            message(FATAL_ERROR
                "Failed to determine the current vision-inference branch with "
                "'git rev-parse --abbrev-ref HEAD': ${GIT_ERROR}")
        endif()

        if("${CURRENT_BRANCH}" STREQUAL "HEAD")
            set(CURRENT_BRANCH "")
        endif()
    endif()

    set(${OUTPUT_VAR} "${CURRENT_BRANCH}" PARENT_SCOPE)
endfunction()

function(determine_shared_dependency_ref OUTPUT_VAR)
    get_current_branch_name(CURRENT_BRANCH)

    if(NOT "${CURRENT_BRANCH}" STREQUAL "")
        if("${CURRENT_BRANCH}" STREQUAL "master")
            set(SELECTED_REF "master")
        else()
            set(SELECTED_REF "develop")
        endif()

        message(STATUS
            "Dependency ref '${SELECTED_REF}' selected from current branch "
            "'${CURRENT_BRANCH}'")
        set(${OUTPUT_VAR} "${SELECTED_REF}" PARENT_SCOPE)
        return()
    endif()

    if(PROJECT_VERSION_RAW MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
        set(SELECTED_REF "master")
    else()
        set(SELECTED_REF "develop")
    endif()

    message(STATUS
        "Dependency ref '${SELECTED_REF}' selected from project version "
        "'${PROJECT_VERSION_RAW}'")
    set(${OUTPUT_VAR} "${SELECTED_REF}" PARENT_SCOPE)
endfunction()

read_versions_from_env()
determine_shared_dependency_ref(SHARED_DEPENDENCY_REF)

foreach(COMPONENT_VERSION_VAR NEURIPLO_VERSION VIDEOCAPTURE_VERSION VISION_CORE_VERSION)
    if(DEFINED ${COMPONENT_VERSION_VAR} AND NOT "${${COMPONENT_VERSION_VAR}}" STREQUAL "" AND
       NOT "${${COMPONENT_VERSION_VAR}}" STREQUAL "${SHARED_DEPENDENCY_REF}")
        message(FATAL_ERROR
            "${COMPONENT_VERSION_VAR}=${${COMPONENT_VERSION_VAR}} does not match "
            "the derived shared dependency ref ${SHARED_DEPENDENCY_REF}. "
            "vision-inference requires neuriplo, videocapture, and vision-core to target the same ref.")
    endif()
endforeach()

set(NEURIPLO_VERSION "${SHARED_DEPENDENCY_REF}" CACHE STRING "neuriplo" FORCE)
set(VIDEOCAPTURE_VERSION "${SHARED_DEPENDENCY_REF}" CACHE STRING "VideoCapture library version" FORCE)
set(VISION_CORE_VERSION "${SHARED_DEPENDENCY_REF}" CACHE STRING "vision-core library version" FORCE)

set(OPENCV_MIN_VERSION "${OPENCV_MIN_VERSION}" CACHE STRING "Minimum OpenCV version")
set(GLOG_MIN_VERSION "${GLOG_MIN_VERSION}" CACHE STRING "Minimum glog version")
set(CMAKE_MIN_VERSION "${CMAKE_MIN_VERSION}" CACHE STRING "Minimum CMake version")

message(STATUS "=== Project Dependency Versions ===")
message(STATUS "shared dependency ref: ${SHARED_DEPENDENCY_REF}")
message(STATUS "neuriplo: ${NEURIPLO_VERSION}")
message(STATUS "VideoCapture: ${VIDEOCAPTURE_VERSION}")
message(STATUS "vision-core: ${VISION_CORE_VERSION}")
message(STATUS "OpenCV Min: ${OPENCV_MIN_VERSION}")
message(STATUS "glog Min: ${GLOG_MIN_VERSION}")
message(STATUS "CMake Min: ${CMAKE_MIN_VERSION}")

# Note: Inference backend versions (ONNX Runtime, TensorRT, LibTorch, etc.)
# are managed by neuriplo, not this project.
