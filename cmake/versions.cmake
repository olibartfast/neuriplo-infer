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
                string(STRIP "${VAR_VALUE}" VAR_VALUE)
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

# Capture any -D overrides from the cmake invocation before reading versions.env,
# because read_versions_from_env() introduces normal variables that would shadow
# the cache entries set via -D in this scope.
set(_OVERRIDE_NEURIPLO_VERSION "${NEURIPLO_VERSION}")
set(_OVERRIDE_VIDEOCAPTURE_VERSION "${VIDEOCAPTURE_VERSION}")
set(_OVERRIDE_VISION_CORE_VERSION "${VISION_CORE_VERSION}")

read_versions_from_env()

# Snapshot the values that came from versions.env (if any) so we can distinguish
# them from the derived shared ref below.
set(_ENV_NEURIPLO_VERSION "${NEURIPLO_VERSION}")
set(_ENV_VIDEOCAPTURE_VERSION "${VIDEOCAPTURE_VERSION}")
set(_ENV_VISION_CORE_VERSION "${VISION_CORE_VERSION}")

determine_shared_dependency_ref(SHARED_DEPENDENCY_REF)

# Resolve each sibling ref independently with the following precedence:
#   1. -D override on the cmake command line
#   2. Value pinned in versions.env (e.g. set on release tags for reproducibility)
#   3. Auto-derived shared ref from branch / VERSION (develop or master)
function(_resolve_sibling_ref VAR_NAME OVERRIDE_VAL ENV_VAL FALLBACK_VAL SOURCE_VAR)
    if(NOT "${OVERRIDE_VAL}" STREQUAL "")
        set(_RESOLVED "${OVERRIDE_VAL}")
        set(${SOURCE_VAR} "cmake -D override" PARENT_SCOPE)
    elseif(NOT "${ENV_VAL}" STREQUAL "")
        set(_RESOLVED "${ENV_VAL}")
        set(${SOURCE_VAR} "versions.env" PARENT_SCOPE)
    else()
        set(_RESOLVED "${FALLBACK_VAL}")
        set(${SOURCE_VAR} "derived (${FALLBACK_VAL})" PARENT_SCOPE)
    endif()
    # Update both the cache (so downstream subdirectories see it) and the
    # parent-scope normal variable (so any local-scope value created earlier
    # by read_versions_from_env() does not shadow the resolved cache value).
    set(${VAR_NAME} "${_RESOLVED}" CACHE STRING "${VAR_NAME}" FORCE)
    set(${VAR_NAME} "${_RESOLVED}" PARENT_SCOPE)
endfunction()

_resolve_sibling_ref(NEURIPLO_VERSION
    "${_OVERRIDE_NEURIPLO_VERSION}" "${_ENV_NEURIPLO_VERSION}"
    "${SHARED_DEPENDENCY_REF}" NEURIPLO_VERSION_SOURCE)
_resolve_sibling_ref(VIDEOCAPTURE_VERSION
    "${_OVERRIDE_VIDEOCAPTURE_VERSION}" "${_ENV_VIDEOCAPTURE_VERSION}"
    "${SHARED_DEPENDENCY_REF}" VIDEOCAPTURE_VERSION_SOURCE)
_resolve_sibling_ref(VISION_CORE_VERSION
    "${_OVERRIDE_VISION_CORE_VERSION}" "${_ENV_VISION_CORE_VERSION}"
    "${SHARED_DEPENDENCY_REF}" VISION_CORE_VERSION_SOURCE)

if(NOT (NEURIPLO_VERSION STREQUAL VIDEOCAPTURE_VERSION AND
        NEURIPLO_VERSION STREQUAL VISION_CORE_VERSION))
    message(WARNING
        "Sibling refs differ: neuriplo=${NEURIPLO_VERSION}, "
        "videocapture=${VIDEOCAPTURE_VERSION}, vision-core=${VISION_CORE_VERSION}. "
        "This is allowed for cross-branch testing but unusual for a release build.")
endif()

set(OPENCV_MIN_VERSION "${OPENCV_MIN_VERSION}" CACHE STRING "Minimum OpenCV version")
set(GLOG_MIN_VERSION "${GLOG_MIN_VERSION}" CACHE STRING "Minimum glog version")
set(CMAKE_MIN_VERSION "${CMAKE_MIN_VERSION}" CACHE STRING "Minimum CMake version")

message(STATUS "=== Project Dependency Versions ===")
message(STATUS "shared dependency ref (fallback): ${SHARED_DEPENDENCY_REF}")
message(STATUS "neuriplo: ${NEURIPLO_VERSION} [source: ${NEURIPLO_VERSION_SOURCE}]")
message(STATUS "VideoCapture: ${VIDEOCAPTURE_VERSION} [source: ${VIDEOCAPTURE_VERSION_SOURCE}]")
message(STATUS "vision-core: ${VISION_CORE_VERSION} [source: ${VISION_CORE_VERSION_SOURCE}]")
message(STATUS "OpenCV Min: ${OPENCV_MIN_VERSION}")
message(STATUS "glog Min: ${GLOG_MIN_VERSION}")
message(STATUS "CMake Min: ${CMAKE_MIN_VERSION}")

# Note: Inference backend versions (ONNX Runtime, TensorRT, LibTorch, etc.)
# are managed by neuriplo, not this project.
