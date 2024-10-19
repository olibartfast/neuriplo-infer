#include "gtest/gtest.h"
#include "DetectorSetup.hpp"

// Helper function to test detector creation
void testDetectorCreation(const std::string& detectorType, const std::type_info& expectedType) {
    auto detector = DetectorSetup::createDetector(detectorType);
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<const void*>(detector.get()) != nullptr);
    EXPECT_EQ(typeid(*detector).name(), expectedType.name());
}

// Test case for YoloV4
TEST(DetectorSetupTest, CreateYoloV4Detector) {
    testDetectorCreation("yolov4", typeid(YoloV4));
}

// Test case for YoloVn (catching multiple versions)
TEST(DetectorSetupTest, CreateYoloVnDetector) {
    testDetectorCreation("yolov5", typeid(YoloVn));
    testDetectorCreation("yolov6", typeid(YoloVn));
    testDetectorCreation("yolov7", typeid(YoloVn));
    testDetectorCreation("yolov8", typeid(YoloVn));
    testDetectorCreation("yolov9", typeid(YoloVn));
}

// Test case for YoloNas
TEST(DetectorSetupTest, CreateYoloNasDetector) {
    testDetectorCreation("yolonas", typeid(YoloNas));
}

// Test case for YOLOv10
TEST(DetectorSetupTest, CreateYOLOv10Detector) {
    testDetectorCreation("yolov10", typeid(YOLOv10));
}

// Test case for RtDetrUltralytics
TEST(DetectorSetupTest, CreateRtDetrUltralyticsDetector) {
    testDetectorCreation("rtdetrul", typeid(RtDetrUltralytics));
}

// Test case for RtDetr
TEST(DetectorSetupTest, CreateRtDetrDetector) {
    testDetectorCreation("rtdetr", typeid(RtDetr));
}

// Test case for unknown type
TEST(DetectorSetupTest, CreateUnknownDetector) {
    auto detector = DetectorSetup::createDetector("unknown");
    EXPECT_EQ(detector, nullptr);
}