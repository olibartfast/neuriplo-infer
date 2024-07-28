#include <gtest/gtest.h>
#include "ObjectDetectionApp.hpp"

TEST(ParseCommandLineArguments, Basic) {
    // Simulate command-line arguments
    const char* argv[] = {
        "program",
        "--type", "yolov5",
        "--source", "input.mp4",
        "--weights", "model.weights",
        "--config", "model.cfg",
        "--labels", "labels.txt",
        "--use-gpu",
        "--min_confidence", "0.5"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    
    AppConfig config = parseCommandLineArguments(argc, const_cast<char**>(argv));

    EXPECT_EQ(config.detectorType, "yolov5");
    EXPECT_EQ(config.source, "input.mp4");
    EXPECT_EQ(config.weights, "model.weights");
    EXPECT_EQ(config.config, "model.cfg");
    EXPECT_EQ(config.labelsPath, "labels.txt");
    EXPECT_TRUE(config.use_gpu);
    EXPECT_FLOAT_EQ(config.confidenceThreshold, 0.5);
}