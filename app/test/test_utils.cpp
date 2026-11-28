#include "AppImageOps.hpp"
#include "TaskRouting.hpp"
#include "utils.hpp"
#include "gtest/gtest.h"
#include <cstring>
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
  // Only the file name carries an extension; a dotted directory does not.
  EXPECT_EQ(getFileExtension("runs.v2/clip"), "");
  EXPECT_EQ(getFileExtension("runs.v2/clip.mp4"), "mp4");
  EXPECT_EQ(getFileExtension(".hidden"), "");
}

// Routing used a case-sensitive substring test for ".jpg"/".png", so
// IMG_0001.JPG or photo.jpeg ran through the video loop and wrote no image.
TEST(UtilsStandalone, IsStillImageSourceIsCaseInsensitiveOnTheFileName) {
  for (const char *path : {"a.jpg", "IMG_0001.JPG", "photo.jpeg", "scan.bmp",
                           "page.tif", "page.TIFF", "x.png", "y.webp"}) {
    EXPECT_TRUE(isStillImageSource(path)) << path;
  }
  for (const char *path : {"clip.mp4", "a.png.mp4", "runs.jpg/clip",
                           "dir.v2/clip", "no_extension", ""}) {
    EXPECT_FALSE(isStillImageSource(path)) << path;
  }
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
  for (const auto *name : {"LightGlue", "lightglue_v2", "superglue",
                           "MatchAnything-EloFTR", "flexglue"}) {
    EXPECT_EQ(getTaskTypeForModel(name),
              neuriplo_tasks::TaskType::KeypointMatching);
  }
  for (const auto *name :
       {"videomae", "pevideo", "v-jepa", "vjepa2-vit-l", "videoprism"}) {
    EXPECT_EQ(getTaskTypeForModel(name),
              neuriplo_tasks::TaskType::VideoClassification);
  }
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
  EXPECT_EQ(getTaskTypeForModel("depth_anything_3"),
            neuriplo_tasks::TaskType::DepthEstimation);
  EXPECT_EQ(getTaskTypeForModel("da3-small"),
            neuriplo_tasks::TaskType::DepthEstimation);
  // Gaze estimation and monocular 3D detection: without sibling support
  // they fail fast naming the family; with it they route to the new tasks.
  // std::invalid_argument surfaces as exit 1 + failure report at every
  // getTaskTypeForModel call site.
  for (const auto *name : {"gaze", "ethxgaze", "gazetr", "urbanomni",
                           "urbanomnidetect2", "monocular3d", "mono3d"}) {
#ifdef NEURIPLO_TASKS_HAS_GAZE_MONO3D
    const auto expected = (std::string(name).find("gaze") != std::string::npos)
                              ? neuriplo_tasks::TaskType::GazeEstimation
                              : neuriplo_tasks::TaskType::Monocular3DDetection;
    EXPECT_EQ(getTaskTypeForModel(name), expected) << name;
#else
    EXPECT_THROW(getTaskTypeForModel(name), std::invalid_argument) << name;
#endif
  }
  EXPECT_EQ(getTaskTypeForModel("groundingdino"),
            neuriplo_tasks::TaskType::OpenVocabDetection);
  EXPECT_EQ(getTaskTypeForModel("lgm-mini"),
            neuriplo_tasks::TaskType::GaussianSplatting);
  EXPECT_EQ(getTaskTypeForModel("grm"),
            neuriplo_tasks::TaskType::GaussianSplatting);
  EXPECT_EQ(getTaskTypeForModel("imageunderstanding"),
            neuriplo_tasks::TaskType::ImageUnderstanding);
}

// neuriplo-tasks v0.8.0 routes any YOLO-prefixed model type containing `depth`
// to its YOLO26 depth family; before the pin bump these fell through to
// Detection and a depth model silently ran as a detector.
TEST(TaskRouting, RoutesYoloDepthFamilyToDepthEstimation) {
  EXPECT_EQ(getTaskTypeForModel("yolo-depth"),
            neuriplo_tasks::TaskType::DepthEstimation);
  EXPECT_EQ(getTaskTypeForModel("yolo26n-depth"),
            neuriplo_tasks::TaskType::DepthEstimation);
  EXPECT_EQ(getTaskTypeForModel("depthanythingv2"),
            neuriplo_tasks::TaskType::DepthEstimation);

  // The seg and pose branches still win over the broader depth match.
  EXPECT_EQ(getTaskTypeForModel("yolo11-seg"),
            neuriplo_tasks::TaskType::InstanceSegmentation);
  EXPECT_EQ(getTaskTypeForModel("vitpose"),
            neuriplo_tasks::TaskType::PoseEstimation);
  EXPECT_EQ(getTaskTypeForModel("yolo26"), neuriplo_tasks::TaskType::Detection);

  // TaskFactory only routes YOLO-prefixed depth types and Depth-Anything-V2;
  // any other name containing "depth" is a detector there, and must be here.
  EXPECT_EQ(getTaskTypeForModel("mydepthnet"),
            neuriplo_tasks::TaskType::Detection);
}

namespace {
int countNonBlack(const neuriplo_tasks::vision::Image &image) {
  int count = 0;
  for (int y = 0; y < image.height(); ++y) {
    const unsigned char *row = image.ptr<unsigned char>(y);
    for (int x = 0; x < image.width(); ++x) {
      if (row[static_cast<size_t>(x) * 3] != 0 ||
          row[static_cast<size_t>(x) * 3 + 1] != 0 ||
          row[static_cast<size_t>(x) * 3 + 2] != 0) {
        ++count;
      }
    }
  }
  return count;
}
} // namespace

TEST(AppImageOpsText, SizeMatchesLayout) {
  using neuriplo_infer::imageops::textSize;
  const auto one = textSize("A", 1);
  EXPECT_EQ(one.x, 5);
  EXPECT_EQ(one.y, 7);
  const auto two = textSize("AB", 1);
  EXPECT_EQ(two.x, 11);
  EXPECT_EQ(two.y, 7);
  const auto scaled = textSize("AB", 2);
  EXPECT_EQ(scaled.x, 22);
  EXPECT_EQ(scaled.y, 14);
  const auto empty = textSize("", 1);
  EXPECT_EQ(empty.x, 0);
  EXPECT_EQ(empty.y, 0);
}

TEST(AppImageOpsText, DrawsPixelsAndClips) {
  using neuriplo_infer::imageops::Color;
  using neuriplo_infer::imageops::drawText;
  using neuriplo_tasks::vision::Image;
  using neuriplo_tasks::vision::PixelType;
  Image image(40, 12, 3, PixelType::UInt8);
  EXPECT_EQ(countNonBlack(image), 0);
  drawText(image, "A", 0, 0, Color{255, 255, 255});
  const int inked = countNonBlack(image);
  EXPECT_GT(inked, 0);
  EXPECT_LT(inked, 40 * 12);
  // Fully off-canvas text is a no-op, not a crash.
  drawText(image, "A", -100, -100, Color{255, 255, 255});
  EXPECT_EQ(countNonBlack(image), inked);
  // Unknown bytes draw nothing.
  Image blank(40, 12, 3, PixelType::UInt8);
  std::string unknown;
  unknown.push_back(static_cast<char>(1));
  unknown.push_back(static_cast<char>(2));
  drawText(blank, unknown, 0, 0, Color{255, 255, 255});
  EXPECT_EQ(countNonBlack(blank), 0);
}

TEST(AppImageOpsText, LowercaseFoldsToUppercase) {
  using neuriplo_infer::imageops::Color;
  using neuriplo_infer::imageops::drawText;
  using neuriplo_tasks::vision::Image;
  using neuriplo_tasks::vision::PixelType;
  Image lower(40, 12, 3, PixelType::UInt8);
  Image upper(40, 12, 3, PixelType::UInt8);
  drawText(lower, "gaze 01", 0, 0, Color{255, 255, 255});
  drawText(upper, "GAZE 01", 0, 0, Color{255, 255, 255});
  EXPECT_GT(countNonBlack(lower), 0);
  EXPECT_EQ(std::memcmp(lower.raw(), upper.raw(), 40 * 12 * 3), 0);
}
