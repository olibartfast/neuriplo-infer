#include <gtest/gtest.h>
#include "ObjectDetectionApp.hpp"
#include <filesystem>

class MockDetector : public Detector {
public:
    std::vector<Detection> postprocess(const std::vector<std::vector<std::any>>& outputs, 
                                       const std::vector<std::vector<int64_t>>& shapes, 
                                       const cv::Size& frame_size) override {
        return {}; // Return mock detection data
    }

    cv::Mat preprocess_image(const cv::Mat& image) override {
        return image; // Return the input image as a mock preprocessing step
    }
};

class MockInference : public InferenceInterface {
public:
    MockInference(const std::string& weights, const std::string& modelConfiguration, bool use_gpu = false)
        : InferenceInterface(weights, modelConfiguration, use_gpu) {}

    std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override {
        return {{{}}, {{}}}; // Return mock inference data
    }
};


TEST(ObjectDetectionApp, Initialization) {
    AppConfig config;
    config.source = "input.mp4";
    config.use_gpu = false;
    config.confidenceThreshold = 0.5;
    config.weights = "model.weights";
    config.labelsPath = "fake_labels.txt";
    std::ofstream labelsFile("fake_labels.txt");
    labelsFile.close();

    config.detectorType = "yolov5";

   // ObjectDetectionApp app(config);

    // Replace actual detector and engine with mocks
    //app.setDetector(std::make_unique<MockDetector>());
    //app.setEngine(std::make_unique<MockInference>(config.weights, config.config, config.use_gpu));

    // You can add more tests to check if the initialization works as expected.
}

TEST(ObjectDetectionApp, RunImageDetection) {
    AppConfig config;
    config.source = "test_image.jpg";
    config.use_gpu = false;
    config.confidenceThreshold = 0.25;
    config.weights = "test_weights.weights";
    config.labelsPath = "test_labels.txt";
    config.detectorType = "yolov5";

    ObjectDetectionApp app(config);

    // Mock objects
    app.setDetector(std::make_unique<MockDetector>());
    app.setEngine(std::make_unique<MockInference>(config.weights, config.config, config.use_gpu));

    // Call the run method and check its behavior
    EXPECT_NO_THROW(app.run());
}