#include <gtest/gtest.h>
#include <fstream>
#include "CommandLineParser.hpp"

// Helper: create a minimal on-disk file so validation passes.
static void touchFile(const char* path) {
    std::ofstream f(path); f.close();
}

TEST(ParseCommandLineArguments, Basic) {
    // Simulate command-line arguments
    const char* argv[] = {
        "program",
        "--type=yolov5",
        "--source=input.mp4",
        "--weights=model.weights",
        "--labels=labels.txt",
        "--use-gpu",
        "--min_confidence=0.5"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    touchFile("input.mp4");
    touchFile("model.weights");
    touchFile("labels.txt");
    AppConfig config = CommandLineParser::parseCommandLineArguments(argc, const_cast<char**>(argv));

    EXPECT_EQ(config.detectorType, "yolov5");
    ASSERT_FALSE(config.sources.empty());
    EXPECT_EQ(config.sources[0], "input.mp4");
    EXPECT_EQ(config.weights, "model.weights");
    EXPECT_EQ(config.labelsPath, "labels.txt");
    EXPECT_TRUE(config.use_gpu);
    EXPECT_FLOAT_EQ(config.confidenceThreshold, 0.5f);
    // Defaults
    EXPECT_FLOAT_EQ(config.nmsThreshold, 0.45f);
    EXPECT_FLOAT_EQ(config.maskThreshold, 0.50f);
}

TEST(ParseCommandLineArguments, ThresholdFlags) {
    const char* argv[] = {
        "program",
        "--type=yoloseg",
        "--source=input.mp4",
        "--weights=model.weights",
        "--min_confidence=0.3",
        "--nms_threshold=0.6",
        "--mask_threshold=0.7"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    touchFile("input.mp4");
    touchFile("model.weights");
    AppConfig config = CommandLineParser::parseCommandLineArguments(argc, const_cast<char**>(argv));

    EXPECT_FLOAT_EQ(config.confidenceThreshold, 0.3f);
    EXPECT_FLOAT_EQ(config.nmsThreshold, 0.6f);
    EXPECT_FLOAT_EQ(config.maskThreshold, 0.7f);
}

TEST(ParseCommandLineArguments, OpenVocabFlags) {
    const char* argv[] = {
        "program",
        "--type=owlv2",
        "--source=input.jpg",
        "--weights=model.onnx",
        "--text_prompts=cat;dog",
        "--tokenizer_vocab=vocab.json",
        "--tokenizer_merges=merges.txt"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    touchFile("input.jpg");
    touchFile("model.onnx");
    touchFile("vocab.json");
    touchFile("merges.txt");

    AppConfig config = CommandLineParser::parseCommandLineArguments(argc, const_cast<char**>(argv));

    EXPECT_EQ(config.detectorType, "owlv2");
    ASSERT_EQ(config.textPrompts.size(), 2u);
    EXPECT_EQ(config.textPrompts[0], "cat");
    EXPECT_EQ(config.textPrompts[1], "dog");
    EXPECT_EQ(config.tokenizerVocabPath, "vocab.json");
    EXPECT_EQ(config.tokenizerMergesPath, "merges.txt");
}

TEST(ParseCommandLineArguments, MultimodalExtraParams) {
    const char* argv[] = {
        "program",
        "--type=gemma4",
        "--source=input.mp4",
        "--weights=model.onnx",
        "--prompt=Summarize the clip",
        "--output_format=JSON",
        "--sample_stride=4",
        "--max_frames=12"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    touchFile("input.mp4");
    touchFile("model.onnx");

    AppConfig config = CommandLineParser::parseCommandLineArguments(argc, const_cast<char**>(argv));

    ASSERT_EQ(config.taskExtraParams.size(), 4u);
    EXPECT_EQ(config.taskExtraParams.at("prompt"), "Summarize the clip");
    EXPECT_EQ(config.taskExtraParams.at("output_format"), "json");
    EXPECT_EQ(config.taskExtraParams.at("sample_stride"), "4");
    EXPECT_EQ(config.taskExtraParams.at("max_frames"), "12");
}
