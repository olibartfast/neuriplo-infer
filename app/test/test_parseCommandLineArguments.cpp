#include "CommandLineParser.hpp"
#include <fstream>
#include <gtest/gtest.h>

// Helper: create a minimal on-disk file so validation passes.
static void touchFile(const char *path) {
  std::ofstream f(path);
  f.close();
}

TEST(ParseCommandLineArguments, Basic) {
  // Simulate command-line arguments
  const char *argv[] = {"program",
                        "--type=yolov5",
                        "--source=input.mp4",
                        "--weights=model.weights",
                        "--labels=labels.txt",
                        "--use-gpu",
                        "--min_confidence=0.5"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.weights");
  touchFile("labels.txt");
  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

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

TEST(ParseCommandLineArguments, CapabilitiesDoesNotRequireRunArguments) {
  const char *argv[] = {"program", "--capabilities"};
  int argc = sizeof(argv) / sizeof(argv[0]);

  const AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_TRUE(config.show_capabilities);
  EXPECT_TRUE(config.sources.empty());
  EXPECT_TRUE(config.weights.empty());
  EXPECT_EQ(config.benchmark_iterations, 10);
  EXPECT_FLOAT_EQ(config.confidenceThreshold, 0.25f);
  EXPECT_EQ(config.batch_size, 1);
}

TEST(ParseCommandLineArguments, ThresholdFlags) {
  const char *argv[] = {"program",
                        "--type=yoloseg",
                        "--source=input.mp4",
                        "--weights=model.weights",
                        "--min_confidence=0.3",
                        "--nms_threshold=0.6",
                        "--mask_threshold=0.7"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.weights");
  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_FLOAT_EQ(config.confidenceThreshold, 0.3f);
  EXPECT_FLOAT_EQ(config.nmsThreshold, 0.6f);
  EXPECT_FLOAT_EQ(config.maskThreshold, 0.7f);
  // Mask stays the default representation when the flag is absent.
  EXPECT_EQ(config.segmentationOutput, "mask");
}

TEST(ParseCommandLineArguments, SegmentationOutputFlag) {
  const char *argv[] = {"program", "--type=yoloseg", "--source=input.mp4",
                        "--weights=model.weights",
                        "--segmentation_output=Polygon"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.weights");
  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_EQ(config.segmentationOutput, "polygon");
}

TEST(ParseCommandLineArguments, InvalidSegmentationOutputExits) {
  const char *argv[] = {"program", "--type=yoloseg", "--source=input.mp4",
                        "--weights=model.weights",
                        "--segmentation_output=contours"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.weights");

  EXPECT_EXIT(CommandLineParser::parseCommandLineArguments(
                  argc, const_cast<char **>(argv)),
              ::testing::ExitedWithCode(1), "--segmentation_output must be");
}

TEST(ParseCommandLineArguments, OpenVocabFlags) {
  const char *argv[] = {"program",
                        "--type=owlv2",
                        "--source=input.jpg",
                        "--weights=model.onnx",
                        "--text_prompts=cat;dog",
                        "--tokenizer_vocab=vocab.json",
                        "--tokenizer_merges=merges.txt"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");
  touchFile("model.onnx");
  touchFile("vocab.json");
  touchFile("merges.txt");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_EQ(config.detectorType, "owlv2");
  ASSERT_EQ(config.textPrompts.size(), 2u);
  EXPECT_EQ(config.textPrompts[0], "cat");
  EXPECT_EQ(config.textPrompts[1], "dog");
  EXPECT_EQ(config.tokenizerVocabPath, "vocab.json");
  EXPECT_EQ(config.tokenizerMergesPath, "merges.txt");
}

TEST(ParseCommandLineArguments, MultimodalExtraParams) {
  const char *argv[] = {"program",
                        "--type=gemma4",
                        "--source=input.mp4",
                        "--weights=model.onnx",
                        "--prompt=Summarize the clip",
                        "--output_format=JSON",
                        "--sample_stride=4",
                        "--max_frames=12"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.onnx");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  ASSERT_EQ(config.taskExtraParams.size(), 4u);
  EXPECT_EQ(config.taskExtraParams.at("prompt"), "Summarize the clip");
  EXPECT_EQ(config.taskExtraParams.at("output_format"), "json");
  EXPECT_EQ(config.taskExtraParams.at("sample_stride"), "4");
  EXPECT_EQ(config.taskExtraParams.at("max_frames"), "12");
}

TEST(ParseCommandLineArguments, ExportMetadataFlag) {
  const char *argv[] = {"program", "--type=yolov5", "--weights=model.weights",
                        "--export_metadata"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("model.weights");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_TRUE(config.export_metadata);
  EXPECT_TRUE(config.sources.empty());
}

TEST(ParseCommandLineArguments, KServeRemoteDoesNotRequireWeights) {
  const char *argv[] = {"program",
                        "--type=yolo26",
                        "--source=input.jpg",
                        "--kserve_endpoint=http://127.0.0.1:8080",
                        "--kserve_model_name=yolo",
                        "--kserve_timeout_ms=5000"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_EQ(config.detectorType, "yolo26");
  EXPECT_EQ(config.kserve_endpoint, "http://127.0.0.1:8080");
  EXPECT_EQ(config.kserve_model_name, "yolo");
  EXPECT_EQ(config.kserve_timeout_ms, 5000);
  EXPECT_TRUE(config.weights.empty());
}

TEST(ParseCommandLineArguments, MissingSourceForVisionTaskExits) {
  // A vision task with no --source (and not --export_metadata) must exit(1)
  // with an actionable message instead of running.
  const char *argv[] = {"program", "--type=yolov5", "--weights=model.weights"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("model.weights");

  EXPECT_EXIT(CommandLineParser::parseCommandLineArguments(
                  argc, const_cast<char **>(argv)),
              ::testing::ExitedWithCode(1), "--source is required");
}

TEST(ParseCommandLineArguments, KServeModelDefaultsToType) {
  const char *argv[] = {"program", "--type=yolo26", "--source=input.jpg",
                        "--kserve_endpoint=http://127.0.0.1:8080"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_EQ(config.kserve_model_name, "yolo26");
}

// --- Server-side ensemble flags ----------------------------------------------

TEST(ParseCommandLineArguments, EnsembleFlagDefaults) {
  const char *argv[] = {"program", "--type=yolo26", "--source=input.jpg",
                        "--weights=model.weights"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");
  touchFile("model.weights");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_EQ(config.input_mode, "preprocessed");
  EXPECT_EQ(config.postprocess_mode, "cpu");
  EXPECT_TRUE(config.task_model.empty());
}

TEST(ParseCommandLineArguments, EncodedImageModeParses) {
  const char *argv[] = {"program",
                        "--type=yolo26",
                        "--source=input.jpg",
                        "--kserve_endpoint=http://127.0.0.1:8080",
                        "--kserve_model_name=yolo_ensemble",
                        "--input_mode=Encoded-Image",
                        "--task_model=yolo",
                        "--postprocess_mode=GPU"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_EQ(config.input_mode, "encoded-image");
  EXPECT_EQ(config.task_model, "yolo");
  EXPECT_EQ(config.postprocess_mode, "gpu");
}

// The ensemble's own metadata describes an encoded image, so without an inner
// model there is nothing to build the task from.
TEST(ParseCommandLineArguments, EncodedImageWithoutTaskModelExits) {
  const char *argv[] = {"program", "--type=yolo26", "--source=input.jpg",
                        "--kserve_endpoint=http://127.0.0.1:8080",
                        "--input_mode=encoded-image"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");

  EXPECT_EXIT(CommandLineParser::parseCommandLineArguments(
                  argc, const_cast<char **>(argv)),
              ::testing::ExitedWithCode(1), "--task_model is required");
}

TEST(ParseCommandLineArguments, EncodedImageWithoutKserveEndpointExits) {
  const char *argv[] = {"program",
                        "--type=yolo26",
                        "--source=input.jpg",
                        "--weights=model.weights",
                        "--input_mode=encoded-image",
                        "--task_model=yolo"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");
  touchFile("model.weights");

  EXPECT_EXIT(CommandLineParser::parseCommandLineArguments(
                  argc, const_cast<char **>(argv)),
              ::testing::ExitedWithCode(1), "requires --kserve_endpoint");
}

// Server-side postprocessing only exists on the ensemble path.
TEST(ParseCommandLineArguments, GpuPostprocessWithoutEncodedImageExits) {
  const char *argv[] = {"program", "--type=yolo26", "--source=input.jpg",
                        "--weights=model.weights", "--postprocess_mode=gpu"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");
  touchFile("model.weights");

  EXPECT_EXIT(CommandLineParser::parseCommandLineArguments(
                  argc, const_cast<char **>(argv)),
              ::testing::ExitedWithCode(1), "--postprocess_mode=gpu requires");
}

TEST(ParseCommandLineArguments, InvalidInputModeExits) {
  const char *argv[] = {"program", "--type=yolo26", "--source=input.jpg",
                        "--weights=model.weights", "--input_mode=telepathy"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");
  touchFile("model.weights");

  EXPECT_EXIT(CommandLineParser::parseCommandLineArguments(
                  argc, const_cast<char **>(argv)),
              ::testing::ExitedWithCode(1), "--input_mode must be");
}

// Warmup and benchmark used to preprocess locally and hand the result to the
// engine, which sends a dense float tensor to an ensemble whose input is an
// encoded UINT8 image. These pin that both flags are accepted in encoded-image
// mode; they route through inferFrame, so the transport is whatever the mode
// selects.
TEST(ParseCommandLineArguments, EncodedImageAcceptsWarmupAndBenchmark) {
  const char *argv[] = {"program",
                        "--type=yolo26seg",
                        "--source=input.jpg",
                        "--kserve_endpoint=http://127.0.0.1:8080",
                        "--kserve_model_name=yolo_ens",
                        "--input_mode=encoded-image",
                        "--task_model=yolo",
                        "--warmup",
                        "--benchmark"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_EQ(config.input_mode, "encoded-image");
  EXPECT_TRUE(config.enable_warmup);
  EXPECT_TRUE(config.enable_benchmark);
}

TEST(ParseCommandLineArguments, TaskModelVersionDefaultsAndParses) {
  const char *argv[] = {"program",
                        "--type=yolo26seg",
                        "--source=input.jpg",
                        "--kserve_endpoint=http://127.0.0.1:8080",
                        "--kserve_model_name=yolo_ens",
                        "--kserve_model_version=3",
                        "--input_mode=encoded-image",
                        "--task_model=yolo",
                        "--task_model_version=7"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  // The inner model's version is independent of the ensemble's.
  EXPECT_EQ(config.kserve_model_version, "3");
  EXPECT_EQ(config.task_model_version, "7");
}

TEST(ParseCommandLineArguments, VideoRunArtifactFlagsDefaultToOff) {
  // A run that asks for neither must behave exactly as before: no CSV path and
  // a preview window, since --no_display is opt-in.
  const char *argv[] = {"program", "--type=yolov5", "--source=input.mp4",
                        "--weights=model.weights", "--labels=labels.txt"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.weights");
  touchFile("labels.txt");
  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_TRUE(config.timings_csv.empty());
  EXPECT_FALSE(config.no_display);
}

TEST(ParseCommandLineArguments, VideoRunArtifactFlagsAreParsed) {
  const char *argv[] = {"program",
                        "--type=yolov5",
                        "--source=input.mp4",
                        "--weights=model.weights",
                        "--labels=labels.txt",
                        "--timings_csv=out/frames.csv",
                        "--no_display"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.weights");
  touchFile("labels.txt");
  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_EQ(config.timings_csv, "out/frames.csv");
  EXPECT_TRUE(config.no_display);
}
