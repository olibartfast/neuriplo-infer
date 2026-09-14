#include "CommandLineParser.hpp"
#include "RunReport.hpp"
#include <filesystem>
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

TEST(ParseCommandLineArguments, MultimodalPromptIsParsed) {
  const char *argv[] = {"program", "--type=gemma4", "--source=input.jpg",
                        "--weights=model.onnx", "--prompt=Describe the image"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");
  touchFile("model.onnx");

  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  ASSERT_EQ(config.taskExtraParams.size(), 1u);
  EXPECT_EQ(config.taskExtraParams.at("prompt"), "Describe the image");
}

// The image understanding task reads only the prompt: video sources ran
// text-only and these flags did nothing, while the run reported success.
TEST(ParseCommandLineArguments, ImageUnderstandingRejectsVideoAndUnusedFlags) {
  touchFile("input.mp4");
  touchFile("input.jpg");
  touchFile("model.onnx");
  {
    const char *argv[] = {"program", "--type=gemma4", "--source=input.mp4",
                          "--weights=model.onnx"};
    int argc = sizeof(argv) / sizeof(argv[0]);
    EXPECT_EXIT(
        {
          AppConfig config = CommandLineParser::parseCommandLineArguments(
              argc, const_cast<char **>(argv));
          (void)config;
        },
        ::testing::ExitedWithCode(1), "one still image");
  }
  {
    const char *argv[] = {"program", "--type=gemma4", "--source=input.jpg",
                          "--weights=model.onnx", "--sample_stride=4"};
    int argc = sizeof(argv) / sizeof(argv[0]);
    EXPECT_EXIT(
        {
          AppConfig config = CommandLineParser::parseCommandLineArguments(
              argc, const_cast<char **>(argv));
          (void)config;
        },
        ::testing::ExitedWithCode(1), "not supported");
  }
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

namespace {
// Parses argv in a child process and expects it to exit 1 with `message`.
void expectRejected(std::vector<const char *> argv, const char *message) {
  argv.insert(argv.begin(), "program");
  const int argc = static_cast<int>(argv.size());
  EXPECT_EXIT(
      {
        AppConfig config = CommandLineParser::parseCommandLineArguments(
            argc, const_cast<char **>(argv.data()));
        (void)config;
      },
      ::testing::ExitedWithCode(1), message);
}
} // namespace

TEST(ParseCommandLineArguments, TimingsCsvIsRejectedForStillImageRuns) {
  touchFile("input.jpg");
  touchFile("model.weights");
  expectRejected({"--type=yolov5", "--source=input.jpg",
                  "--weights=model.weights", "--timings_csv=t.csv"},
                 "--timings_csv applies to video");
  expectRejected({"--type=yolov5", "--weights=model.weights",
                  "--export_metadata", "--timings_csv=t.csv"},
                 "--timings_csv applies to video");
}

TEST(ParseCommandLineArguments, WarmupAndBenchmarkAreRejectedForVideoRuns) {
  touchFile("input.mp4");
  touchFile("model.weights");
  expectRejected({"--type=yolov5", "--source=input.mp4",
                  "--weights=model.weights", "--benchmark"},
                 "apply to single still-image runs");
}

TEST(ParseCommandLineArguments, TaskModelRequiresEncodedImageMode) {
  touchFile("input.mp4");
  expectRejected({"--type=yolov5", "--source=input.mp4",
                  "--kserve_endpoint=http://127.0.0.1:8080",
                  "--task_model=yolo"},
                 "--task_model only applies");
}

TEST(ParseCommandLineArguments,
     ExplicitNmsThresholdIsRejectedWithGpuPostprocess) {
  touchFile("input.jpg");
  expectRejected({"--type=yolo26seg", "--source=input.jpg",
                  "--kserve_endpoint=http://127.0.0.1:8080",
                  "--input_mode=encoded-image", "--task_model=yolo",
                  "--postprocess_mode=gpu", "--nms_threshold=0.3"},
                 "the ensemble applies its own");
}

// cv::CommandLineParser records a conversion error only when a value is read;
// checking before the reads let --batch=abc through as batch 0.
TEST(ParseCommandLineArguments, MalformedNumericValueIsRejected) {
  touchFile("input.mp4");
  touchFile("model.weights");
  expectRejected({"--type=yolov5", "--source=input.mp4",
                  "--weights=model.weights", "--batch=abc"},
                 "");
}

TEST(ParseCommandLineArguments, UnknownOptionIsRejected) {
  touchFile("input.mp4");
  touchFile("model.weights");
  expectRejected({"--type=yolov5", "--source=input.mp4",
                  "--weights=model.weights", "--output-video=out.mp4"},
                 "Unknown option");
}

TEST(ParseCommandLineArguments, OutputThatWouldOverwriteTheSourceIsRejected) {
  touchFile("input.mp4");
  touchFile("model.weights");
  expectRejected({"--type=yolov5", "--source=input.mp4",
                  "--weights=model.weights", "--timings_csv=./input.mp4"},
                 "would overwrite the source");
}

TEST(ParseCommandLineArguments, RecordsWhetherSegmentationOutputWasGiven) {
  touchFile("input.mp4");
  touchFile("model.weights");
  {
    const char *argv[] = {"program", "--type=yoloseg", "--source=input.mp4",
                          "--weights=model.weights"};
    int argc = sizeof(argv) / sizeof(argv[0]);
    const AppConfig config = CommandLineParser::parseCommandLineArguments(
        argc, const_cast<char **>(argv));
    EXPECT_FALSE(config.segmentation_output_explicit);
  }
  {
    const char *argv[] = {"program", "--type=yoloseg", "--source=input.mp4",
                          "--weights=model.weights", "--so=polygon"};
    int argc = sizeof(argv) / sizeof(argv[0]);
    const AppConfig config = CommandLineParser::parseCommandLineArguments(
        argc, const_cast<char **>(argv));
    EXPECT_TRUE(config.segmentation_output_explicit);
    EXPECT_EQ(config.segmentationOutput, "polygon");
  }
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

TEST(ParseCommandLineArguments, OutputVideoDefaultsToEmpty) {
  const char *argv[] = {"program", "--type=yolov5", "--source=input.mp4",
                        "--weights=model.weights", "--labels=labels.txt"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.weights");
  touchFile("labels.txt");
  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  EXPECT_TRUE(config.output_video.empty());
}

TEST(ParseCommandLineArguments, OutputVideoIsRejectedForStillImageSources) {
  const char *argv[] = {"program",
                        "--type=yolov5",
                        "--source=input.jpg",
                        "--weights=model.weights",
                        "--labels=labels.txt",
                        "--output_video=out.mp4"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.jpg");
  touchFile("model.weights");
  touchFile("labels.txt");
  EXPECT_EXIT(
      {
        AppConfig config = CommandLineParser::parseCommandLineArguments(
            argc, const_cast<char **>(argv));
        (void)config;
      },
      ::testing::ExitedWithCode(1), "applies to video sources only");
}

#ifndef VIDEOCAPTURE_WITH_WRITER
TEST(ParseCommandLineArguments, OutputVideoRequiresWriterBuild) {
  const char *argv[] = {"program",
                        "--type=yolov5",
                        "--source=input.mp4",
                        "--weights=model.weights",
                        "--labels=labels.txt",
                        "--output_video=out.mp4"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.weights");
  touchFile("labels.txt");
  EXPECT_EXIT(
      {
        AppConfig config = CommandLineParser::parseCommandLineArguments(
            argc, const_cast<char **>(argv));
        (void)config;
      },
      ::testing::ExitedWithCode(1), "NEURIPLO_INFER_WITH_VIDEOWRITER");
}
#endif

#ifdef VIDEOCAPTURE_WITH_WRITER
TEST(ParseCommandLineArguments, OutputVideoIsParsedForVideoSources) {
  const char *argv[] = {"program",
                        "--type=yolov5",
                        "--source=input.mp4",
                        "--weights=model.weights",
                        "--labels=labels.txt",
                        "--output_video=out.mp4"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  touchFile("model.weights");
  touchFile("labels.txt");
  AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

  // Reader-side frames are written fresh; the parse path only records the
  // destination.
  EXPECT_EQ(config.output_video, "out.mp4");
}
#endif

TEST(ParseCommandLineArguments, OutputVideoIsRejectedForMetadataExport) {
  // Metadata export never enters a frame loop, so no file would be written.
  const char *argv[] = {"program", "--type=yolov5", "--weights=model.weights",
                        "--export_metadata", "--output_video=out.mp4"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("model.weights");
  EXPECT_EXIT(
      {
        AppConfig config = CommandLineParser::parseCommandLineArguments(
            argc, const_cast<char **>(argv));
        (void)config;
      },
      ::testing::ExitedWithCode(1), "video inference runs only");
}

TEST(ParseCommandLineArguments, OutputVideoIsRejectedForTextTasks) {
  // Image understanding is dispatched before any frame loop, so it never
  // writes a video either.
  const char *argv[] = {"program", "--type=gemma4", "--weights=model.weights",
                        "--prompt=describe", "--output_video=out.mp4"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("model.weights");
  EXPECT_EXIT(
      {
        AppConfig config = CommandLineParser::parseCommandLineArguments(
            argc, const_cast<char **>(argv));
        (void)config;
      },
      ::testing::ExitedWithCode(1), "video inference runs only");
}

TEST(ParseCommandLineArguments, HelpDoesNotWriteAFailedRunReport) {
  const auto report = std::filesystem::temp_directory_path() /
                      "neuriplo-infer-help-run-report.json";
  std::filesystem::remove(report);
  const char *argv[] = {"program", "--help"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  // main() arms the configuration report before parsing; the help exit must
  // not leave it armed.
  EXPECT_EXIT(
      {
        neuriplo_infer::armConfigurationExitReport(report);
        AppConfig config = CommandLineParser::parseCommandLineArguments(
            argc, const_cast<char **>(argv));
        (void)config;
      },
      ::testing::ExitedWithCode(1), "");
  EXPECT_FALSE(std::filesystem::exists(report));
}

TEST(ParseCommandLineArguments, KserveTransportDefaultsToACompiledTransport) {
  const char *argv[] = {"program", "--type=yolov5", "--source=input.mp4",
                        "--kserve_endpoint=http://127.0.0.1:8080"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  const AppConfig config = CommandLineParser::parseCommandLineArguments(
      argc, const_cast<char **>(argv));

#ifdef KSERVE_CLIENT_WITH_GRPC
  EXPECT_EQ(config.kserve_transport, "grpc");
#else
  EXPECT_EQ(config.kserve_transport, "http");
#endif
}

#if defined(NEURIPLO_INFER_WITH_KSERVE) && !defined(KSERVE_CLIENT_WITH_GRPC)
TEST(ParseCommandLineArguments, GrpcTransportIsRejectedWithoutGrpcSupport) {
  // Accepting it used to build an HTTP client silently.
  const char *argv[] = {"program", "--type=yolov5", "--source=input.mp4",
                        "--kserve_endpoint=http://127.0.0.1:8080",
                        "--kserve_transport=grpc"};
  int argc = sizeof(argv) / sizeof(argv[0]);
  touchFile("input.mp4");
  EXPECT_EXIT(
      {
        AppConfig config = CommandLineParser::parseCommandLineArguments(
            argc, const_cast<char **>(argv));
        (void)config;
      },
      ::testing::ExitedWithCode(1), "needs a build with gRPC");
}
#endif
