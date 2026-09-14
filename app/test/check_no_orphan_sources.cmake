# Fails when a file in app/src is part of no build at all.
#
# Two such files accumulated after the command refactor, holding whole second
# copies of the processing and rendering paths. Nothing referenced them and
# nothing compiled them, so they drifted until they no longer built against
# their own headers -- while still reading, to anyone opening the directory,
# like the code that runs.
#
# The check is deliberately insensitive to build configuration: a source named
# anywhere in app/CMakeLists.txt counts as claimed, including one added only
# under a condition such as KServe. Being unreachable in this build is normal;
# being unreferenced in every build is the defect.
#
# Run as: cmake -DAPP_DIR=<app> -P this.cmake

if(NOT DEFINED APP_DIR)
    message(FATAL_ERROR "APP_DIR is required")
endif()

get_filename_component(APP_DIR "${APP_DIR}" ABSOLUTE)

file(READ "${APP_DIR}/CMakeLists.txt" build_file)
file(GLOB sources RELATIVE "${APP_DIR}/src" "${APP_DIR}/src/*.cpp")

# Finding nothing means the directory was not where this was told to look, and
# a check that inspects no files must not report that everything is fine.
if(NOT sources)
    message(FATAL_ERROR "no sources found under ${APP_DIR}/src")
endif()

set(orphans "")
foreach(source IN LISTS sources)
    string(FIND "${build_file}" "${source}" position)
    if(position EQUAL -1)
        list(APPEND orphans "${source}")
    endif()
endforeach()

if(orphans)
    string(REPLACE ";" "\n  " listing "${orphans}")
    message(FATAL_ERROR
        "app/src holds sources no build compiles:\n  ${listing}\n"
        "Add them to APP_LIB_SOURCES in app/CMakeLists.txt, or delete them. "
        "A file that compiles nowhere cannot be kept honest by any other gate.")
endif()

list(LENGTH sources count)
message(STATUS "all ${count} sources in app/src are claimed by the build")
