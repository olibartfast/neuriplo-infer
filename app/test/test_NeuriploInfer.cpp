#include "AppConfig.hpp"
#include "ResultRenderer.hpp"
#include "neuriplo/tasks/core/task_config.hpp"
#include "neuriplo/tasks/core/vision/image.hpp"
#include <gtest/gtest.h>

// Verify that AppConfig threshold fields map correctly to
// neuriplo_tasks::TaskConfig. This mirrors the mapping done in
// InferencePipelineBuilder task config mapping.

TEST(NeuriploInferTaskConfig, ThresholdMapping) {
  AppConfig config;
  config.confidenceThreshold = 0.3f;
  config.nmsThreshold = 0.6f;
  config.maskThreshold = 0.7f;

  neuriplo_tasks::TaskConfig task_config;
  task_config.confidence_threshold = config.confidenceThreshold;
  task_config.nms_threshold = config.nmsThreshold;
  task_config.mask_threshold = config.maskThreshold;

  EXPECT_FLOAT_EQ(task_config.confidence_threshold, 0.3f);
  EXPECT_FLOAT_EQ(task_config.nms_threshold, 0.6f);
  EXPECT_FLOAT_EQ(task_config.mask_threshold, 0.7f);
}

TEST(NeuriploInferTaskConfig, DefaultThresholdsRoundtrip) {
  AppConfig config;
  config.confidenceThreshold = 0.25f;
  // nmsThreshold and maskThreshold use in-class defaults

  neuriplo_tasks::TaskConfig task_config;
  task_config.confidence_threshold = config.confidenceThreshold;
  task_config.nms_threshold = config.nmsThreshold;
  task_config.mask_threshold = config.maskThreshold;

  EXPECT_FLOAT_EQ(task_config.confidence_threshold, 0.25f);
  EXPECT_FLOAT_EQ(task_config.nms_threshold, 0.45f);
  EXPECT_FLOAT_EQ(task_config.mask_threshold, 0.50f);
}

TEST(NeuriploInferTaskConfig, OpenVocabFieldsRoundtrip) {
  AppConfig config;
  config.textPrompts = {"cat", "dog"};
  config.tokenizerVocabPath = "vocab.json";
  config.tokenizerMergesPath = "merges.txt";

  neuriplo_tasks::TaskConfig task_config;
  task_config.text_prompts = config.textPrompts;

  EXPECT_EQ(task_config.text_prompts.size(), 2u);
  EXPECT_EQ(task_config.text_prompts[0], "cat");
  EXPECT_EQ(task_config.text_prompts[1], "dog");
  EXPECT_EQ(config.tokenizerVocabPath, "vocab.json");
  EXPECT_EQ(config.tokenizerMergesPath, "merges.txt");
}

TEST(NeuriploInferTaskConfig, ExtraParamsRoundtrip) {
  AppConfig config;
  config.taskExtraParams = {{"prompt", "Describe the image"},
                            {"output_format", "json"},
                            {"sample_stride", "4"},
                            {"max_frames", "8"}};

  neuriplo_tasks::TaskConfig task_config;
  task_config.extra_params = config.taskExtraParams;

  ASSERT_EQ(task_config.extra_params.size(), 4u);
  EXPECT_EQ(task_config.extra_params.at("prompt"), "Describe the image");
  EXPECT_EQ(task_config.extra_params.at("output_format"), "json");
  EXPECT_EQ(task_config.extra_params.at("sample_stride"), "4");
  EXPECT_EQ(task_config.extra_params.at("max_frames"), "8");
}

TEST(ClassificationRendering, KeepsAllRankedResultsVisible) {
  using neuriplo_tasks::vision::Image;
  using neuriplo_tasks::vision::PixelType;
  Image image = Image::zeros(320, 240, 3, PixelType::UInt8);
  std::vector<neuriplo_tasks::Result> results;
  const std::vector<std::string> classes{"malamute", "Eskimo dog",
                                         "Siberian husky", "Norwegian elkhound",
                                         "German shepherd"};
  for (size_t i = 0; i < classes.size(); ++i) {
    neuriplo_tasks::Classification value;
    value.class_id = static_cast<float>(i);
    value.class_confidence = 0.9f - 0.1f * i;
    results.emplace_back(value);
  }
  DefaultResultRenderer renderer;
  renderer.render(
      results, image,
      RenderContext{neuriplo_tasks::TaskType::Classification, classes, 0.25f});
  auto rowHasInk = [&](int y) {
    const std::uint8_t *row = image.ptr<std::uint8_t>(y);
    for (int x = 0; x < image.width() * image.channels(); ++x) {
      if (row[x] != 0) {
        return true;
      }
    }
    return false;
  };
  int text_rows = 0;
  bool in_text = false;
  for (int y = 0; y < image.height(); ++y) {
    const bool ink = rowHasInk(y);
    if (ink && !in_text) {
      ++text_rows;
    }
    in_text = ink;
  }
  EXPECT_EQ(text_rows, 5);
  // Bands occupy the left half; the rightmost column stays empty.
  for (int y = 0; y < image.height(); ++y) {
    const std::uint8_t *row = image.ptr<std::uint8_t>(y);
    const int last = (image.width() - 1) * image.channels();
    EXPECT_EQ(row[last], 0);
    EXPECT_EQ(row[last + 1], 0);
    EXPECT_EQ(row[last + 2], 0);
  }
}

TEST(KeypointMatchingRendering, AppliesStrictThresholdAndTargetOffset) {
  using neuriplo_tasks::vision::Image;
  using neuriplo_tasks::vision::PixelType;
  Image image = Image::zeros(100, 40, 3, PixelType::UInt8);
  neuriplo_tasks::KeypointMatching pair;
  neuriplo_tasks::Keypoint a, b;
  a.x = 10;
  a.y = 10;
  b.x = 20;
  b.y = 10;
  pair.source_keypoints.push_back(a);
  pair.target_keypoints.push_back(b);
  pair.matches.emplace_back(0, 0, 0.2f);
  DefaultResultRenderer renderer;
  const std::vector<std::string> classes;
  const RenderContext context{neuriplo_tasks::TaskType::KeypointMatching,
                              classes, 0.2f, 50};
  auto countNonZero = [&](const Image &img) {
    std::size_t n = 0;
    for (int y = 0; y < img.height(); ++y) {
      const std::uint8_t *row = img.ptr<std::uint8_t>(y);
      for (int x = 0; x < img.width() * img.channels(); ++x) {
        if (row[x] != 0) {
          ++n;
        }
      }
    }
    return n;
  };
  auto countInRect = [&](const Image &img, int rx, int ry, int rw, int rh) {
    std::size_t n = 0;
    for (int y = ry; y < ry + rh; ++y) {
      const std::uint8_t *row = img.ptr<std::uint8_t>(y);
      for (int x = rx; x < rx + rw; ++x) {
        for (int c = 0; c < img.channels(); ++c) {
          if (row[x * img.channels() + c] != 0) {
            ++n;
          }
        }
      }
    }
    return n;
  };
  renderer.render({pair}, image, context);
  EXPECT_EQ(countNonZero(image), 0u);
  pair.matches[0].score = 0.3f;
  renderer.render({pair}, image, context);
  EXPECT_GT(countInRect(image, 67, 7, 7, 7), 0u);
  EXPECT_EQ(countInRect(image, 17, 20, 7, 7), 0u);
}
