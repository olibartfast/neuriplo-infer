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

TEST(TaskRouting, MirrorsNeuriploTasksContractAliases) {
  EXPECT_EQ(getTaskTypeForModel("yolo26"), neuriplo_tasks::TaskType::Detection);
  EXPECT_EQ(getTaskTypeForModel("rtdetrultralytics"),
            neuriplo_tasks::TaskType::Detection);
  EXPECT_EQ(getTaskTypeForModel("edgecrafter-seg"),
            neuriplo_tasks::TaskType::InstanceSegmentation);
  EXPECT_EQ(getTaskTypeForModel("ecpose-small"),
            neuriplo_tasks::TaskType::PoseEstimation);
  EXPECT_EQ(getTaskTypeForModel("resnet50"),
            neuriplo_tasks::TaskType::Classification);
  EXPECT_EQ(getTaskTypeForModel("my_tensorflow_model"),
            neuriplo_tasks::TaskType::Classification);
  EXPECT_EQ(getTaskTypeForModel("depth_anything_v2"),
            neuriplo_tasks::TaskType::DepthEstimation);
  EXPECT_EQ(getTaskTypeForModel("groundingdino"),
            neuriplo_tasks::TaskType::OpenVocabDetection);
  EXPECT_EQ(getTaskTypeForModel("lgm-mini"),
            neuriplo_tasks::TaskType::GaussianSplatting);
  EXPECT_EQ(getTaskTypeForModel("grm"),
            neuriplo_tasks::TaskType::GaussianSplatting);
  EXPECT_EQ(getTaskTypeForModel("imageunderstanding"),
            neuriplo_tasks::TaskType::ImageUnderstanding);
}
