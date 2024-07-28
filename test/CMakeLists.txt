enable_testing()

include(GoogleTest)

# Add test executable
add_executable(runUnitTests
    ${CMAKE_CURRENT_LIST_DIR}/test_DetectorSetup.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/test_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_parseCommandLineArguments.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_ObjectDetectionApp.cpp    
)

target_link_libraries(runUnitTests gtest gtest_main)

# Discover and run tests
gtest_discover_tests(runUnitTests)