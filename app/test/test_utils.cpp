#include "TaskRouting.hpp"
#include "utils.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>

// Test Fixture for setting up temporary files and directories
class UtilsTest : public ::testing::Test {
protected:
  std::string tempFilePath;
  std::string tempDirPath;

  void SetUp() override {
    // Create a temporary directory and file
    tempDirPath = "./temp_test_dir";
    tempFilePath = tempDirPath + "/tempfile.txt";

    // Create directory
    std::filesystem::create_directory(tempDirPath);

    // Create a temporary file with some content
    std::ofstream tempFile(tempFilePath);
    tempFile << "Some content"; // Write some content to the file
    tempFile.close();

    // Create label file for ReadLabelNames test
    std::ofstream labelFile(tempDirPath + "/templabels.txt");
    for (int i = 0; i < 80; ++i) {
      labelFile << "label_" << i << "\n"; // Create dummy labels
    }
    labelFile.close();
  }

  void TearDown() override {
    // Remove the temporary file and directory
    std::filesystem::remove_all(tempDirPath);
  }
};

TEST_F(UtilsTest, IsDirectory) {
  EXPECT_TRUE(isDirectory(tempDirPath));
  EXPECT_FALSE(isDirectory("not_a_directory"));
}

TEST_F(UtilsTest, IsFile) {
  EXPECT_TRUE(isFile(tempFilePath));
  EXPECT_FALSE(isFile("not_a_file"));
}

TEST_F(UtilsTest, GetFileExtension) {
  EXPECT_EQ(getFileExtension("image.png"), "png");
  EXPECT_EQ(getFileExtension("archive.tar.gz"), "gz");
  EXPECT_EQ(getFileExtension("no_extension"), "");
}

TEST_F(UtilsTest, ReadLabelNames) {
  auto labels = readLabelNames(tempDirPath + "/templabels.txt");
  EXPECT_EQ(labels.size(), 80);    // Replace with expected size
  EXPECT_EQ(labels[0], "label_0"); // Replace with expected label (note newline)
}

TEST(UtilsStandalone, NormalizeModelType) {
  EXPECT_EQ(normalizeModelType("OWL-ViT"), "owlvit");
  EXPECT_EQ(normalizeModelType(" Gemma_4 "), "gemma4");
  EXPECT_EQ(normalizeModelType("Video Understanding"), "videounderstanding");
}

TEST(TaskRouting, MirrorsVisionCoreContractAliases) {
  EXPECT_EQ(getTaskTypeForModel("yolo26"), vision_core::TaskType::Detection);
  EXPECT_EQ(getTaskTypeForModel("rtdetrultralytics"),
            vision_core::TaskType::Detection);
  EXPECT_EQ(getTaskTypeForModel("edgecrafter-seg"),
            vision_core::TaskType::InstanceSegmentation);
  EXPECT_EQ(getTaskTypeForModel("ecpose-small"),
            vision_core::TaskType::PoseEstimation);
  EXPECT_EQ(getTaskTypeForModel("resnet50"),
            vision_core::TaskType::Classification);
  EXPECT_EQ(getTaskTypeForModel("my_tensorflow_model"),
            vision_core::TaskType::Classification);
  EXPECT_EQ(getTaskTypeForModel("depth_anything_v2"),
            vision_core::TaskType::DepthEstimation);
  EXPECT_EQ(getTaskTypeForModel("groundingdino"),
            vision_core::TaskType::OpenVocabDetection);
  EXPECT_EQ(getTaskTypeForModel("lgm-mini"),
            vision_core::TaskType::GaussianSplatting);
  EXPECT_EQ(getTaskTypeForModel("grm"),
            vision_core::TaskType::GaussianSplatting);
  EXPECT_EQ(getTaskTypeForModel("imageunderstanding"),
            vision_core::TaskType::ImageUnderstanding);
}
