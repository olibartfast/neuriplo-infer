#include "gtest/gtest.h"
#include "DetectorSetup.hpp"
#include "YoloV4.hpp"
#include "YoloVn.hpp"
#include "YOLOv10.hpp"
#include "YoloNas.hpp"
#include "RtDetr.hpp"
#include "RtDetrUltralytics.hpp"

// Test case for YoloV4
TEST(DetectorSetupTest, CreateYoloV4Detector) {
    auto detector = createDetector("yolov4");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloV4*>(detector.get()) != nullptr);
}

// Test case for YoloVn (catching multiple versions)
TEST(DetectorSetupTest, CreateYoloVnDetector) {
    auto detector = createDetector("yolov5");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);

    detector = createDetector("yolov6");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);

    detector = createDetector("yolov7");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);

    detector = createDetector("yolov8");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);

    detector = createDetector("yolov9");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloVn*>(detector.get()) != nullptr);
}

// Test case for YoloNas
TEST(DetectorSetupTest, CreateYoloNasDetector) {
    auto detector = createDetector("yolonas");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YoloNas*>(detector.get()) != nullptr);
}

// Test case for YOLOv10
TEST(DetectorSetupTest, CreateYOLOv10Detector) {
    auto detector = createDetector("yolov10");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<YOLOv10*>(detector.get()) != nullptr);
}

// Test case for RtDetrUltralytics
TEST(DetectorSetupTest, CreateRtDetrUltralyticsDetector) {
    auto detector = createDetector("rtdetrul");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<RtDetrUltralytics*>(detector.get()) != nullptr);
}

// Test case for RtDetr
TEST(DetectorSetupTest, CreateRtDetrDetector) {
    auto detector = createDetector("rtdetr");
    EXPECT_NE(detector, nullptr);
    EXPECT_TRUE(dynamic_cast<RtDetr*>(detector.get()) != nullptr);
}

// Test case for unknown type
TEST(DetectorSetupTest, CreateUnknownDetector) {
    auto detector = createDetector("unknown");
    EXPECT_EQ(detector, nullptr);
}
