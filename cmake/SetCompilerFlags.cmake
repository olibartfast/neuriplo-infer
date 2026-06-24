if(NOT TARGET ${PROJECT_NAME})
    message(WARNING "SetCompilerFlags.cmake expected target ${PROJECT_NAME} to exist")
    return()
endif()

if(CMAKE_CUDA_COMPILER)
    # CUDA builds keep separable compilation enabled and use the same CUDA flag
    # on every platform that supports CUDA.
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
endif()

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Debug>:/Od>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Debug>:-O0>
    )
endif()

message("CMake target compile options configured for ${PROJECT_NAME}")
