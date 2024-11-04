# Add Google Test
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

# Add the source files for the tests
set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/test_main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_DetectorSetup.cpp
)

# Create an executable for the tests
add_executable(runDetectorsTests ${TEST_SOURCES})

target_include_directories(runDetectorsTests PRIVATE
${CMAKE_SOURCE_DIR}/common
${CMAKE_SOURCE_DIR}/detectors/inc
${OpenCV_INCLUDE_DIRS}
${CMAKE_SOURCE_DIR}/detectors/inc
${CMAKE_SOURCE_DIR}/detectors/src
${CMAKE_SOURCE_DIR}/detectors/src/models
)

# Link the executable with the detectors library and GTest
target_link_libraries(runDetectorsTests PRIVATE
    detectors
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Add the test to CTest
add_test(NAME DetectorsTests COMMAND runDetectorsTests)

