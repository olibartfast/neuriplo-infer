# Optional compile-speed helpers. No behavior change unless tools are present.

option(ENABLE_CCACHE "Use ccache when available" ON)

if(ENABLE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "" FORCE)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "" FORCE)
        message(STATUS "Compile speed: using ccache at ${CCACHE_PROGRAM}")
    endif()
endif()
