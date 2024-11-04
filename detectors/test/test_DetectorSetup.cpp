#include "gtest/gtest.h"
#include "DetectorSetup.hpp"


// Test case for YoloV4
TEST(DetectorSetupTest, CreateYoloV4Detector) {
    auto detector = DetectorSetup::createDetector("yolov4");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloV4*>(detector.get()) != nullptr);
}

// Test case for YoloVn (catching multiple versions)
TEST(DetectorSetupTest, CreateYoloVnDetector) {
    auto detector = DetectorSetup::createDetector("yolov5");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);

    detector = DetectorSetup::createDetector("yolov6");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);

    detector = DetectorSetup::createDetector("yolov7");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);

    detector = DetectorSetup::createDetector("yolov8");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);

    detector = DetectorSetup::createDetector("yolov9");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);
}

// Test case for YoloNas
TEST(DetectorSetupTest, CreateYoloNasDetector) {
    auto detector = DetectorSetup::createDetector("yolonas");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloNas*>(detector.get()) != nullptr);
}

// Test case for YOLOv10
TEST(DetectorSetupTest, CreateYOLOv10Detector) {
    auto detector = DetectorSetup::createDetector("yolov10");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YOLOv10*>(detector.get()) != nullptr);
}

// Test case for RtDetrUltralytics
TEST(DetectorSetupTest, CreateRtDetrUltralyticsDetector) {
    auto detector = DetectorSetup::createDetector("rtdetrul");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<RtDetrUltralytics*>(detector.get()) != nullptr);
}

// Test case for RtDetr
TEST(DetectorSetupTest, CreateRtDetrDetector) {
    auto detector = DetectorSetup::createDetector("rtdetr");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<RtDetr*>(detector.get()) != nullptr);
}

// Test case for unknown type
TEST(DetectorSetupTest, CreateUnknownDetector) {
    // Expect an exception to be thrown when an unknown detector type is passed
    EXPECT_THROW(
        {
            try {
                auto detector = DetectorSetup::createDetector("unknown");
            } catch (const std::invalid_argument& e) {
                // Ensure the exception message is correct (optional)
                EXPECT_STREQ(e.what(), "Unknown detector type");
                throw;
            }
        },
        std::invalid_argument
    );
}
