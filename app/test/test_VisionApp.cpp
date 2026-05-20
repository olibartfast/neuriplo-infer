#include <gtest/gtest.h>
#include "AppConfig.hpp"
#include "vision-core/core/task_config.hpp"

// Verify that AppConfig threshold fields map correctly to vision_core::TaskConfig.
// This mirrors the mapping done in VisionApp::VisionApp().

TEST(VisionAppTaskConfig, ThresholdMapping) {
    AppConfig config;
    config.confidenceThreshold = 0.3f;
    config.nmsThreshold        = 0.6f;
    config.maskThreshold       = 0.7f;

    vision_core::TaskConfig task_config;
    task_config.confidence_threshold = config.confidenceThreshold;
    task_config.nms_threshold        = config.nmsThreshold;
    task_config.mask_threshold       = config.maskThreshold;

    EXPECT_FLOAT_EQ(task_config.confidence_threshold, 0.3f);
    EXPECT_FLOAT_EQ(task_config.nms_threshold,        0.6f);
    EXPECT_FLOAT_EQ(task_config.mask_threshold,       0.7f);
}

TEST(VisionAppTaskConfig, DefaultThresholdsRoundtrip) {
    AppConfig config;
    config.confidenceThreshold = 0.25f;
    // nmsThreshold and maskThreshold use in-class defaults

    vision_core::TaskConfig task_config;
    task_config.confidence_threshold = config.confidenceThreshold;
    task_config.nms_threshold        = config.nmsThreshold;
    task_config.mask_threshold       = config.maskThreshold;

    EXPECT_FLOAT_EQ(task_config.confidence_threshold, 0.25f);
    EXPECT_FLOAT_EQ(task_config.nms_threshold,        0.45f);
    EXPECT_FLOAT_EQ(task_config.mask_threshold,       0.50f);
}

TEST(VisionAppTaskConfig, OpenVocabFieldsRoundtrip) {
    AppConfig config;
    config.textPrompts = {"cat", "dog"};
    config.tokenizerVocabPath = "vocab.json";
    config.tokenizerMergesPath = "merges.txt";

    vision_core::TaskConfig task_config;
    task_config.text_prompts = config.textPrompts;

    EXPECT_EQ(task_config.text_prompts.size(), 2u);
    EXPECT_EQ(task_config.text_prompts[0], "cat");
    EXPECT_EQ(task_config.text_prompts[1], "dog");
    EXPECT_EQ(config.tokenizerVocabPath, "vocab.json");
    EXPECT_EQ(config.tokenizerMergesPath, "merges.txt");
}

TEST(VisionAppTaskConfig, ExtraParamsRoundtrip) {
    AppConfig config;
    config.taskExtraParams = {
        {"prompt", "Describe the image"},
        {"output_format", "json"},
        {"sample_stride", "4"},
        {"max_frames", "8"}
    };

    vision_core::TaskConfig task_config;
    task_config.extra_params = config.taskExtraParams;

    ASSERT_EQ(task_config.extra_params.size(), 4u);
    EXPECT_EQ(task_config.extra_params.at("prompt"), "Describe the image");
    EXPECT_EQ(task_config.extra_params.at("output_format"), "json");
    EXPECT_EQ(task_config.extra_params.at("sample_stride"), "4");
    EXPECT_EQ(task_config.extra_params.at("max_frames"), "8");
}
