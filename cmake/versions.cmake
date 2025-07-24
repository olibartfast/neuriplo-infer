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

# External C++ Libraries (fetched via CMake FetchContent)
set(INFERENCE_ENGINES_VERSION ${INFERENCE_ENGINES_VERSION} CACHE STRING "InferenceEngines library version")
set(VIDEOCAPTURE_VERSION ${VIDEOCAPTURE_VERSION} CACHE STRING "VideoCapture library version")

# System Dependencies (minimum versions)
set(OPENCV_MIN_VERSION ${OPENCV_MIN_VERSION} CACHE STRING "Minimum OpenCV version")
set(GLOG_MIN_VERSION ${GLOG_MIN_VERSION} CACHE STRING "Minimum glog version")
set(CMAKE_MIN_VERSION ${CMAKE_MIN_VERSION} CACHE STRING "Minimum CMake version")

# Print version information for debugging
message(STATUS "=== Project Dependency Versions ===")
message(STATUS "InferenceEngines: ${INFERENCE_ENGINES_VERSION}")
message(STATUS "VideoCapture: ${VIDEOCAPTURE_VERSION}")
message(STATUS "OpenCV Min: ${OPENCV_MIN_VERSION}")
message(STATUS "glog Min: ${GLOG_MIN_VERSION}")
message(STATUS "CMake Min: ${CMAKE_MIN_VERSION}")

# Note: Inference backend versions (ONNX Runtime, TensorRT, LibTorch, etc.)
# are managed by the InferenceEngines library, not this project.
# See the InferenceEngines library for backend-specific version management. 
