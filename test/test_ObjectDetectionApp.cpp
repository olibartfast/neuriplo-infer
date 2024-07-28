#include <gtest/gtest.h>
#include "ObjectDetectionApp.hpp"

class MockDetector : public Detector {
public:
    std::vector<Detection> postprocess(...) override {
        return {}; // Return mock detection data
    }
    // Implement other virtual methods as needed
};

class MockEngine : public InferenceEngine {
public:
    std::pair<std::vector<cv::Mat>, std::vector<cv::Size>> get_infer_results(...) override {
        return {{}, {}}; // Return mock inference data
    }
    // Implement other virtual methods as needed
};

TEST(ObjectDetectionApp, Initialization) {
    AppConfig config;
    config.source = "input.mp4";
    config.use_gpu = true;
    config.confidenceThreshold = 0.5;
    config.config = "model.cfg";
    config.weights = "model.weights";
    config.labelsPath = "labels.txt";
    config.detectorType = "yolov5";

    ObjectDetectionApp app(config);

    // Replace actual detector and engine with mocks
    app.detector = std::make_shared<MockDetector>();
    app.engine = std::make_shared<MockEngine>();

    // You can add more tests to check if the initialization works as expected.
}

TEST(ObjectDetectionApp, RunImageDetection) {
    AppConfig config;
    config.source = "test_image.jpg";
    config.use_gpu = false;
    config.confidenceThreshold = 0.25;
    config.config = "test_config.cfg";
    config.weights = "test_weights.weights";
    config.labelsPath = "test_labels.txt";
    config.detectorType = "yolov5";

    ObjectDetectionApp app(config);

    // Mock objects
    auto mockDetector = std::make_shared<MockDetector>();
    auto mockEngine = std::make_shared<MockEngine>();
    app.detector = mockDetector;
    app.engine = mockEngine;

    // Call the run method and check its behavior
    EXPECT_NO_THROW(app.run());
}
