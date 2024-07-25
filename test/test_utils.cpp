#include "gtest/gtest.h"
#include "utils.hpp"
#include <filesystem>

TEST(UtilsTest, IsDirectory) {
    EXPECT_TRUE(isDirectory("/home"));
    EXPECT_FALSE(isDirectory("not_a_directory"));
}

TEST(UtilsTest, IsFile) {
    EXPECT_TRUE(isFile("/home/yourfile.txt"));
    EXPECT_FALSE(isFile("not_a_file"));
}

TEST(UtilsTest, GetFileExtension) {
    EXPECT_EQ(getFileExtension("image.png"), "png");
    EXPECT_EQ(getFileExtension("archive.tar.gz"), "gz");
    EXPECT_EQ(getFileExtension("no_extension"), "");
}

TEST(UtilsTest, ReadLabelNames) {
    auto labels = readLabelNames("path/to/labels.txt");
    EXPECT_EQ(labels.size(), expected_size); // Replace with expected size
    EXPECT_EQ(labels[0], "expected_label"); // Replace with expected label
}

TEST(UtilsTest, DrawLabel) {
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
    draw_label(img, "test_label", 0.95, 10, 10);
    // Optionally, add checks to verify the label was drawn
    // This part is more visual and might be hard to test with assertions
}