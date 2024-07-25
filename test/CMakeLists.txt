enable_testing()

include(GoogleTest)

# Add test executable
add_executable(runUnitTests
    test_DetectorSetup.cpp
    test_InferenceBackendSetup.cpp
    test_utils.cpp
    test_VideoCaptureFactory.cpp
)

target_link_libraries(runUnitTests gtest gtest_main)

# Discover and run tests
gtest_discover_tests(runUnitTests)