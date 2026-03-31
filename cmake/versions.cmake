# Centralized version management for this project's dependencies
# This file should be the single source of truth for versions managed by this project

# Function to read versions from versions.env file
function(read_versions_from_env)
    set(VERSIONS_ENV_FILE "${CMAKE_CURRENT_SOURCE_DIR}/versions.env")
    
    if(NOT EXISTS ${VERSIONS_ENV_FILE})
        message(FATAL_ERROR "versions.env file not found at ${VERSIONS_ENV_FILE}")
    endif()
    
    # Read the file and parse each line
    file(READ ${VERSIONS_ENV_FILE} VERSIONS_CONTENT)
    string(REPLACE "\n" ";" VERSIONS_LINES "${VERSIONS_CONTENT}")
    
    foreach(LINE ${VERSIONS_LINES})
        # Skip empty lines and comments
        if(LINE AND NOT LINE MATCHES "^#")
            # Extract variable name and value
            string(REGEX MATCH "^([A-Z_]+)=(.+)$" MATCH "${LINE}")
            if(MATCH)
                set(VAR_NAME "${CMAKE_MATCH_1}")
                set(VAR_VALUE "${CMAKE_MATCH_2}")
                # Remove quotes if present
                string(REGEX REPLACE "^\"(.*)\"$" "\\1" VAR_VALUE "${VAR_VALUE}")
                # Set the variable
                set(${VAR_NAME} "${VAR_VALUE}" PARENT_SCOPE)
            endif()
        endif()
    endforeach()
endfunction()

# Read versions from the .env file
read_versions_from_env()

if(NOT DEFINED DEPENDENCIES_VERSION OR "${DEPENDENCIES_VERSION}" STREQUAL "")
    message(FATAL_ERROR "DEPENDENCIES_VERSION must be set in versions.env")
endif()

function(validate_current_branch_matches_dependencies_ref)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
        message(STATUS "Skipping branch/ref validation because ${CMAKE_SOURCE_DIR} is not a git checkout")
        return()
    endif()

    set(IS_ALLOWED_RELEASE_BRANCH FALSE)

    function(check_branch_ref_match BRANCH_NAME)
        if("${BRANCH_NAME}" STREQUAL "${DEPENDENCIES_VERSION}")
            set(BRANCH_MATCH_RESULT TRUE PARENT_SCOPE)
            return()
        endif()

        if("${DEPENDENCIES_VERSION}" STREQUAL "develop" AND
           ("${BRANCH_NAME}" STREQUAL "master" OR "${BRANCH_NAME}" MATCHES "^release/"))
            message(STATUS
                "Allowing release branch '${BRANCH_NAME}' to use DEPENDENCIES_VERSION="
                "'${DEPENDENCIES_VERSION}'")
            set(BRANCH_MATCH_RESULT TRUE PARENT_SCOPE)
            return()
        endif()

        if("${DEPENDENCIES_VERSION}" STREQUAL "develop" AND
           "${BRANCH_NAME}" MATCHES "^(feat|fix|refactor|docs|chore)/")
            message(STATUS
                "Allowing topic branch '${BRANCH_NAME}' to use DEPENDENCIES_VERSION="
                "'${DEPENDENCIES_VERSION}'")
            set(BRANCH_MATCH_RESULT TRUE PARENT_SCOPE)
            return()
        endif()

        set(BRANCH_MATCH_RESULT FALSE PARENT_SCOPE)
    endfunction()

    set(CURRENT_BRANCH "")

    # On pull_request workflows, the base branch is the compatibility target.
    if(DEFINED ENV{GITHUB_BASE_REF} AND NOT "$ENV{GITHUB_BASE_REF}" STREQUAL "")
        check_branch_ref_match("$ENV{GITHUB_BASE_REF}")
        if(BRANCH_MATCH_RESULT)
            return()
        endif()
    endif()

    # GitHub Actions checks out pull requests in detached HEAD state.
    # Prefer the explicit branch name from the workflow environment when present.
    if(DEFINED ENV{GITHUB_HEAD_REF} AND NOT "$ENV{GITHUB_HEAD_REF}" STREQUAL "")
        set(CURRENT_BRANCH "$ENV{GITHUB_HEAD_REF}")
    elseif(DEFINED ENV{GITHUB_REF_NAME} AND NOT "$ENV{GITHUB_REF_NAME}" STREQUAL "")
        set(CURRENT_BRANCH "$ENV{GITHUB_REF_NAME}")
    endif()

    if(NOT "${CURRENT_BRANCH}" STREQUAL "")
        check_branch_ref_match("${CURRENT_BRANCH}")
        if(NOT BRANCH_MATCH_RESULT)
            message(FATAL_ERROR
                "Current vision-inference branch '${CURRENT_BRANCH}' does not match "
                "DEPENDENCIES_VERSION='${DEPENDENCIES_VERSION}'. "
                "Check out the '${DEPENDENCIES_VERSION}' branch, use a release branch, "
                "or update versions.env.")
        endif()
        return()
    endif()

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
        message(FATAL_ERROR
            "vision-inference is in a detached HEAD state, but DEPENDENCIES_VERSION="
            "${DEPENDENCIES_VERSION} requires a matching branch checkout.")
    endif()

    check_branch_ref_match("${CURRENT_BRANCH}")
    if(NOT BRANCH_MATCH_RESULT)
        message(FATAL_ERROR
            "Current vision-inference branch '${CURRENT_BRANCH}' does not match "
            "DEPENDENCIES_VERSION='${DEPENDENCIES_VERSION}'. "
            "Check out the '${DEPENDENCIES_VERSION}' branch, use a release branch, "
            "or update versions.env.")
    endif()
endfunction()

validate_current_branch_matches_dependencies_ref()

foreach(COMPONENT_VERSION_VAR NEURIPLO_VERSION VIDEOCAPTURE_VERSION VISION_CORE_VERSION)
    if(DEFINED ${COMPONENT_VERSION_VAR} AND NOT "${${COMPONENT_VERSION_VAR}}" STREQUAL "" AND
       NOT "${${COMPONENT_VERSION_VAR}}" STREQUAL "${DEPENDENCIES_VERSION}")
        message(FATAL_ERROR
            "${COMPONENT_VERSION_VAR}=${${COMPONENT_VERSION_VAR}} does not match "
            "DEPENDENCIES_VERSION=${DEPENDENCIES_VERSION}. "
            "vision-inference requires neuriplo, videocapture, and vision-core to target the same ref.")
    endif()
endforeach()

# External C++ Libraries (fetched via CMake FetchContent)
set(DEPENDENCIES_VERSION ${DEPENDENCIES_VERSION} CACHE STRING
    "Shared git ref for neuriplo, VideoCapture, and vision-core" FORCE)
set(NEURIPLO_VERSION ${DEPENDENCIES_VERSION} CACHE STRING "neuriplo" FORCE)
set(VIDEOCAPTURE_VERSION ${DEPENDENCIES_VERSION} CACHE STRING "VideoCapture library version" FORCE)
set(VISION_CORE_VERSION ${DEPENDENCIES_VERSION} CACHE STRING "vision-core library version" FORCE)

# System Dependencies (minimum versions)
set(OPENCV_MIN_VERSION ${OPENCV_MIN_VERSION} CACHE STRING "Minimum OpenCV version")
set(GLOG_MIN_VERSION ${GLOG_MIN_VERSION} CACHE STRING "Minimum glog version")
set(CMAKE_MIN_VERSION ${CMAKE_MIN_VERSION} CACHE STRING "Minimum CMake version")

# Print version information for debugging
message(STATUS "=== Project Dependency Versions ===")
message(STATUS "shared dependency ref: ${DEPENDENCIES_VERSION}")
message(STATUS "neuriplo: ${NEURIPLO_VERSION}")
message(STATUS "VideoCapture: ${VIDEOCAPTURE_VERSION}")
message(STATUS "vision-core: ${VISION_CORE_VERSION}")
message(STATUS "OpenCV Min: ${OPENCV_MIN_VERSION}")
message(STATUS "glog Min: ${GLOG_MIN_VERSION}")
message(STATUS "CMake Min: ${CMAKE_MIN_VERSION}")

# Note: Inference backend versions (ONNX Runtime, TensorRT, LibTorch, etc.)
# are managed by the neuriplo, not this project.
# See the neuriplo library for backend-specific version management. 
