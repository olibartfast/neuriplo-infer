#include "CLICommands.hpp"
#include "InferencePipeline.hpp"
#include "RunReport.hpp"

#include "neuriplo/tasks/core/task_interface.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>

/**
 * Proves the instrumentation is wired where the work happens.
 *
 * The RunReport unit tests cover the document; these cover the placement of
 * the timers, using fakes so a successful run is reachable without a model or
 * a compiled backend. Placement is the part that silently rots: a timer in the
 * wrong scope still produces a plausible-looking report.
 */

namespace {

using json = nlohmann::json;
using neuriplo_infer::RunReport;
using neuriplo_infer::RunStage;

class FakeEngine : public InferenceInterface {
public:
  FakeEngine() : InferenceInterface("fake-weights", false, 1) {
    inference_metadata_.addInput("images", {1, 3, 64, 64}, 1);
    inference_metadata_.addOutput("output", {1, 8}, 1);
  }

  std::tuple<std::vector<std::vector<TensorElement>>,
             std::vector<std::vector<int64_t>>>
  get_infer_results(const std::vector<std::vector<uint8_t>> &inputs) override {
    ++calls;
    payloads.push_back(inputs);
    if (delay.count() > 0) {
      std::this_thread::sleep_for(delay);
    }
    std::vector<TensorElement> values(8, TensorElement{0.0F});
    return {{values}, {{1, 8}}};
  }

  int calls{0};
  /// Every request's input tensors, in call order.
  std::vector<std::vector<std::vector<uint8_t>>> payloads;
  /// Makes an ignored call cost measurable time, so a test can tell whether
  /// that time was counted.
  std::chrono::milliseconds delay{0};
};

class FakeTask : public neuriplo_tasks::TaskInterface {
public:
  explicit FakeTask(const neuriplo_tasks::ModelInfo &model_info)
      : TaskInterface(model_info) {}

  neuriplo_tasks::TaskType getTaskType() override {
    return neuriplo_tasks::TaskType::Detection;
  }

  std::vector<std::vector<uint8_t>>
  preprocess(const std::vector<neuriplo_tasks::Image> & /*images*/) override {
    ++preprocess_calls;
    return {{0, 1, 2, 3}};
  }

  std::vector<neuriplo_tasks::Result> postprocess(
      const neuriplo_tasks::Size & /*frame_size*/,
      const std::vector<neuriplo_tasks::Tensor> & /*tensors*/) override {
    ++postprocess_calls;
    return {};
  }

  int preprocess_calls{0};
  int postprocess_calls{0};
};

class CountingRenderer : public ResultRenderer {
public:
  void render(const std::vector<neuriplo_tasks::Result> & /*results*/,
              cv::Mat & /*image*/, const RenderContext & /*context*/) override {
    ++calls;
  }

  int calls{0};
};

class RunDiagnostics : public ::testing::Test {
protected:
  void SetUp() override {
    previous_directory_ = std::filesystem::current_path();
    directory_ = std::filesystem::temp_directory_path() /
                 "neuriplo-run-diagnostics-test";
    std::filesystem::remove_all(directory_);
    std::filesystem::create_directories(directory_);

    source_ = directory_ / "fixture.png";
    ASSERT_TRUE(cv::imwrite(source_.string(),
                            cv::Mat(64, 64, CV_8UC3, cv::Scalar(20, 90, 160))));

    // The binary writes its output relative to the working directory, and so
    // does the report.
    std::filesystem::current_path(directory_);
  }

  void TearDown() override {
    std::filesystem::current_path(previous_directory_);
    std::filesystem::remove_all(directory_);
  }

  InferencePipeline makePipeline(RunReport &report) {
    InferencePipeline pipeline;
    pipeline.config.detectorType = "fake";
    pipeline.config.sources = {source_.string()};
    pipeline.config.batch_size = 1;
    pipeline.report = &report;

    auto engine = std::make_unique<FakeEngine>();
    pipeline.inference_metadata = engine->get_inference_metadata();
    pipeline.engine = std::move(engine);

    neuriplo_tasks::ModelInfo model_info;
    model_info.addInput("images", {1, 3, 64, 64}, 1);
    model_info.addOutput("output", {1, 8}, 1);
    pipeline.model_info = model_info;
    pipeline.task = std::make_unique<FakeTask>(model_info);
    pipeline.task_type = neuriplo_tasks::TaskType::Detection;
    pipeline.renderer = std::make_unique<CountingRenderer>();
    return pipeline;
  }

  json report() const {
    std::ifstream stream(directory_ / RunReport::kDefaultPath);
    EXPECT_TRUE(stream.is_open()) << "no run report was written";
    return json::parse(stream);
  }

  std::filesystem::path previous_directory_;
  std::filesystem::path directory_;
  std::filesystem::path source_;
};

TEST_F(RunDiagnostics, ASuccessfulImageRunTimesEveryStageItRan) {
  RunReport collected;
  auto pipeline = makePipeline(collected);

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  const json document = report();
  EXPECT_EQ(document.at("status"), "success");
  EXPECT_TRUE(document.at("error").is_null());

  const json metrics = document.at("metrics");
  EXPECT_EQ(metrics.at("samples"), 1);
  // A still image is one sample and no frames.
  EXPECT_TRUE(metrics.at("frames").is_null());

  const json stages = metrics.at("stages_ms");
  for (const char *stage :
       {"source", "preprocess", "inference", "postprocess", "render"}) {
    if (std::string(stage) == "source") {
      continue; // Source is attributed, not timed.
    }
    EXPECT_FALSE(stages.at(stage).is_null()) << stage << " was not timed";
    EXPECT_GE(stages.at(stage).get<double>(), 0.0) << stage;
  }
  // Nothing loaded a model in this pipeline, so that stage stays absent.
  EXPECT_TRUE(stages.at("model_load").is_null());
  EXPECT_FALSE(metrics.at("throughput_per_second").is_null());
}

TEST_F(RunDiagnostics, RenderTimeExcludesInferenceAndViceVersa) {
  RunReport collected;
  auto pipeline = makePipeline(collected);

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  const json stages = report().at("metrics").at("stages_ms");
  const double total = stages.at("preprocess").get<double>() +
                       stages.at("inference").get<double>() +
                       stages.at("postprocess").get<double>() +
                       stages.at("render").get<double>();

  // Overlapping timers would make the parts exceed the whole.
  EXPECT_LE(total, report().at("metrics").at("wall_time_ms").get<double>());
}

TEST_F(RunDiagnostics, AnUnreadableSourceFailsAtTheSourceStage) {
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {(directory_ / "missing.png").string()};

  try {
    RunInferenceCommand().execute(pipeline);
    FAIL() << "an unreadable source must not run";
  } catch (const std::exception &e) {
    collected.fail(collected.currentStage(), e.what());
  }
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  const json document = report();
  EXPECT_EQ(document.at("status"), "failed");
  EXPECT_EQ(document.at("error").at("stage"), "source");
  EXPECT_EQ(document.at("metrics").at("samples"), 0);
  // Nothing ran, so no stage after the source may claim a measurement.
  const json stages = document.at("metrics").at("stages_ms");
  EXPECT_TRUE(stages.at("inference").is_null());
  EXPECT_TRUE(stages.at("render").is_null());
}

TEST_F(RunDiagnostics, OpticalFlowTimesItsOwnStages) {
  // Optical flow never goes through inferFrame, so it is the case where a
  // timer in one place silently leaves another path unmeasured.
  const auto second = directory_ / "fixture-b.png";
  ASSERT_TRUE(cv::imwrite(second.string(),
                          cv::Mat(64, 64, CV_8UC3, cv::Scalar(160, 90, 20))));

  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {source_.string(), second.string()};
  pipeline.task_type = neuriplo_tasks::TaskType::OpticalFlow;

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  const json metrics = report().at("metrics");
  EXPECT_EQ(metrics.at("samples"), 1) << "a processed pair is one sample";
  const json stages = metrics.at("stages_ms");
  for (const char *stage :
       {"preprocess", "inference", "postprocess", "render"}) {
    EXPECT_FALSE(stages.at(stage).is_null()) << stage << " was not timed";
  }
}

TEST_F(RunDiagnostics, ImageUnderstandingTimesItsOwnStages) {
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.task_type = neuriplo_tasks::TaskType::ImageUnderstanding;

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  const json metrics = report().at("metrics");
  EXPECT_EQ(metrics.at("samples"), 1);
  const json stages = metrics.at("stages_ms");
  for (const char *stage :
       {"preprocess", "inference", "postprocess", "render"}) {
    EXPECT_FALSE(stages.at(stage).is_null()) << stage << " was not timed";
  }
}

TEST_F(RunDiagnostics, AnUnreadableOpticalFlowPairFailsRatherThanSkips) {
  // Skipping the pair returned 0 and wrote status "success" with no samples
  // and no artifact, which a consumer cannot tell apart from a run that had
  // nothing to do.
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {source_.string(),
                             (directory_ / "missing.png").string()};
  pipeline.task_type = neuriplo_tasks::TaskType::OpticalFlow;

  try {
    RunInferenceCommand().execute(pipeline);
    FAIL() << "an unreadable half of the pair must not report success";
  } catch (const std::exception &e) {
    collected.fail(collected.currentStage(), e.what());
  }
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  const json document = report();
  EXPECT_EQ(document.at("status"), "failed");
  EXPECT_EQ(document.at("error").at("stage"), "source");
  EXPECT_EQ(document.at("metrics").at("samples"), 0);
}

TEST_F(RunDiagnostics, AFailedOpticalFlowRunIsAttributedToTheStageItRanIn) {
  const auto second = directory_ / "fixture-b.png";
  ASSERT_TRUE(cv::imwrite(second.string(),
                          cv::Mat(64, 64, CV_8UC3, cv::Scalar(160, 90, 20))));

  class ThrowingTask : public FakeTask {
  public:
    using FakeTask::FakeTask;
    std::vector<std::vector<uint8_t>>
    preprocess(const std::vector<neuriplo_tasks::Image> & /*images*/) override {
      throw std::runtime_error("preprocess refused the pair");
    }
  };

  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {source_.string(), second.string()};
  pipeline.task_type = neuriplo_tasks::TaskType::OpticalFlow;
  neuriplo_tasks::ModelInfo model_info;
  model_info.addInput("images", {1, 3, 64, 64}, 1);
  model_info.addOutput("output", {1, 8}, 1);
  pipeline.task = std::make_unique<ThrowingTask>(model_info);

  try {
    RunInferenceCommand().execute(pipeline);
    FAIL() << "the pipeline must not swallow the failure";
  } catch (const std::exception &e) {
    collected.fail(collected.currentStage(), e.what());
  }
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  // Without a timer on this path the stage would still read model_load.
  EXPECT_EQ(report().at("error").at("stage"), "preprocess");
}

TEST_F(RunDiagnostics, WarmupAndBenchmarkTimeStaysOutOfTheStageTotals) {
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.enable_warmup = true; // five ignored inferences
  pipeline.config.enable_benchmark = true;
  pipeline.config.benchmark_iterations = 3;

  auto *engine = dynamic_cast<FakeEngine *>(pipeline.engine.get());
  ASSERT_NE(engine, nullptr);
  engine->delay = std::chrono::milliseconds(10);

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  const json metrics = report().at("metrics");
  EXPECT_EQ(engine->calls, 9) << "warmup and benchmark did run";
  EXPECT_EQ(metrics.at("samples"), 1);
  // Eight of the nine calls produced no sample. Counting their 80 ms would
  // inflate the stage total and divide the sample count by it.
  EXPECT_LT(metrics.at("stages_ms").at("inference").get<double>(), 60.0);
}

TEST_F(RunDiagnostics, APipelineWithoutACollectorStillRuns) {
  RunReport unused;
  auto pipeline = makePipeline(unused);
  pipeline.report = nullptr;

  EXPECT_EQ(RunInferenceCommand().execute(pipeline), 0);
}

// Writes a short MJPG clip the OpenCV capture backend reads back.
std::filesystem::path writeFixtureVideo(const std::filesystem::path &path,
                                        int frames) {
  cv::VideoWriter writer(path.string(),
                         cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0,
                         cv::Size(64, 64));
  EXPECT_TRUE(writer.isOpened()) << "could not create " << path;
  for (int i = 0; i < frames; ++i) {
    writer.write(cv::Mat(64, 64, CV_8UC3, cv::Scalar(i * 20, 90, 160)));
  }
  return path;
}

int countVideoFrames(const std::filesystem::path &path) {
  cv::VideoCapture capture(path.string());
  int frames = 0;
  cv::Mat frame;
  while (capture.read(frame)) {
    ++frames;
  }
  return frames;
}

TEST_F(RunDiagnostics, AHeadlessVideoRunWritesOneTimingsRowPerFrame) {
  const auto video = writeFixtureVideo(directory_ / "fixture.avi", 5);
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {video.string()};
  pipeline.config.no_display = true; // the test has no screen to open one on
  pipeline.config.timings_csv = (directory_ / "out" / "timings.csv").string();

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  std::ifstream csv(pipeline.config.timings_csv);
  ASSERT_TRUE(csv.is_open()) << "parent directories were not created";
  std::string line;
  ASSERT_TRUE(std::getline(csv, line));
  EXPECT_EQ(line, "frame,latency_us");
  int rows = 0;
  while (std::getline(csv, line)) {
    EXPECT_EQ(line.rfind(std::to_string(rows) + ",", 0), 0U) << line;
    ++rows;
  }
  EXPECT_EQ(rows, 5);

  const json metrics = report().at("metrics");
  EXPECT_EQ(metrics.at("samples"), 1);
  EXPECT_EQ(metrics.at("frames"), 5);
}

TEST_F(RunDiagnostics, ATimingsCsvThatCannotBeWrittenFailsTheRun) {
  if (!std::filesystem::exists("/dev/full")) {
    GTEST_SKIP() << "needs /dev/full to simulate a full disk";
  }
  const auto video = writeFixtureVideo(directory_ / "fixture.avi", 3);
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {video.string()};
  pipeline.config.no_display = true;
  // Opens fine, fails on flush: the truncated-CSV case.
  pipeline.config.timings_csv = "/dev/full";

  EXPECT_THROW(RunInferenceCommand().execute(pipeline), std::runtime_error);
}

TEST_F(RunDiagnostics, AnImageThatCannotBeSavedFailsInsteadOfCountingASample) {
  // Learn the output name from a successful run, then occupy both the primary
  // and the /tmp fallback destination with directories so neither write can
  // succeed.
  {
    RunReport probe_report;
    auto probe = makePipeline(probe_report);
    ASSERT_EQ(RunInferenceCommand().execute(probe), 0);
  }
  std::string name;
  for (const auto &entry : std::filesystem::directory_iterator("data/output")) {
    if (entry.path().extension() == ".png") {
      name = entry.path().filename().string();
    }
  }
  ASSERT_FALSE(name.empty()) << "the probe run wrote no image";
  const auto primary = std::filesystem::path("data/output") / name;
  const auto fallback = std::filesystem::path("/tmp/neuriplo-infer-" + name);
  std::filesystem::remove(primary);
  std::filesystem::create_directories(primary);
  std::filesystem::remove_all(fallback);
  std::filesystem::create_directories(fallback);

  RunReport collected;
  auto pipeline = makePipeline(collected);
  bool threw = false;
  try {
    RunInferenceCommand().execute(pipeline);
  } catch (const std::exception &e) {
    threw = true;
    collected.fail(collected.currentStage(), e.what());
  }
  std::filesystem::remove_all(fallback);
  ASSERT_TRUE(threw) << "a run with no saved image must not succeed";
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  const json document = report();
  EXPECT_EQ(document.at("status"), "failed");
  EXPECT_EQ(document.at("error").at("stage"), "render");
  EXPECT_EQ(document.at("metrics").at("samples"), 0);
}

TEST_F(RunDiagnostics, ClassificationCountsEachSourceFrameOnce) {
  // Overlapping windows used to count every frame of every window:
  // 4 * (6 - 4 + 1) = 12 frames for a 6-frame clip.
  const auto video = writeFixtureVideo(directory_ / "fixture.avi", 6);
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {video.string()};
  pipeline.config.no_display = true;
  pipeline.config.num_frames = 4;
  pipeline.task_type = neuriplo_tasks::TaskType::VideoClassification;

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);

  EXPECT_EQ(dynamic_cast<FakeEngine *>(pipeline.engine.get())->calls, 3);
  EXPECT_EQ(report().at("metrics").at("frames"), 6);
}

#ifdef NEURIPLO_INFER_WITH_KSERVE
TEST_F(RunDiagnostics, EncodedWarmupAndBenchmarkSendTheSourceFileBytes) {
  // Warmup and benchmark used to re-encode the decoded frame, so they
  // measured a payload of different content and size than the real request.
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.encoded_image = true;
  pipeline.config.enable_warmup = true;
  pipeline.config.enable_benchmark = true;
  pipeline.config.benchmark_iterations = 2;

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);

  std::ifstream stream(source_, std::ios::binary);
  const std::vector<uint8_t> file_bytes(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  const auto *engine = dynamic_cast<FakeEngine *>(pipeline.engine.get());
  ASSERT_NE(engine, nullptr);
  // Five warmup iterations, the real request, two benchmark iterations.
  ASSERT_EQ(engine->payloads.size(), 8U);
  for (const auto &payload : engine->payloads) {
    ASSERT_EQ(payload.size(), 1U);
    EXPECT_EQ(payload[0], file_bytes);
  }
}
#endif

#ifdef VIDEOCAPTURE_WITH_WRITER
TEST_F(RunDiagnostics, OutputVideoCreatesMissingParentDirectories) {
  const auto video = writeFixtureVideo(directory_ / "fixture.avi", 5);
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {video.string()};
  pipeline.config.no_display = true;
  pipeline.config.output_video =
      (directory_ / "nested" / "deeper" / "annotated.avi").string();

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);

  EXPECT_EQ(countVideoFrames(pipeline.config.output_video), 5);
}

TEST_F(RunDiagnostics, AVideoWithNoFramesFailsInsteadOfSkippingTheOutput) {
  const auto video = writeFixtureVideo(directory_ / "empty.avi", 0);
  if (!cv::VideoCapture(video.string()).isOpened()) {
    GTEST_SKIP() << "this OpenCV build cannot open a zero-frame clip";
  }
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {video.string()};
  pipeline.config.no_display = true;
  pipeline.config.output_video = (directory_ / "annotated.avi").string();

  EXPECT_THROW(RunInferenceCommand().execute(pipeline), std::runtime_error);
  neuriplo_infer::writeRunReport(collected, RunReport::kDefaultPath);
  EXPECT_EQ(report().at("metrics").at("samples"), 0);
}

TEST_F(RunDiagnostics, AVideoRunWritesEveryFrameToTheOutputVideo) {
  const auto video = writeFixtureVideo(directory_ / "fixture.avi", 5);
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {video.string()};
  pipeline.config.no_display = true;
  pipeline.config.output_video = (directory_ / "annotated.avi").string();

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);

  EXPECT_EQ(countVideoFrames(pipeline.config.output_video), 5);
}

TEST_F(RunDiagnostics, AClassificationVideoShorterThanItsWindowStillWritesIt) {
  const auto video = writeFixtureVideo(directory_ / "fixture.avi", 3);
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {video.string()};
  pipeline.config.no_display = true;
  pipeline.config.num_frames = 16; // longer than the clip: no window closes
  pipeline.config.output_video = (directory_ / "annotated.avi").string();
  pipeline.task_type = neuriplo_tasks::TaskType::VideoClassification;

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);

  EXPECT_EQ(dynamic_cast<FakeEngine *>(pipeline.engine.get())->calls, 0);
  EXPECT_EQ(countVideoFrames(pipeline.config.output_video), 3);
}

TEST_F(RunDiagnostics, AClassificationVideoWritesEachSourceFrameExactlyOnce) {
  const auto video = writeFixtureVideo(directory_ / "fixture.avi", 6);
  RunReport collected;
  auto pipeline = makePipeline(collected);
  pipeline.config.sources = {video.string()};
  pipeline.config.no_display = true;
  pipeline.config.num_frames = 4; // windows close on frames 4, 5 and 6
  pipeline.config.output_video = (directory_ / "annotated.avi").string();
  pipeline.task_type = neuriplo_tasks::TaskType::VideoClassification;

  ASSERT_EQ(RunInferenceCommand().execute(pipeline), 0);

  EXPECT_EQ(dynamic_cast<FakeEngine *>(pipeline.engine.get())->calls, 3);
  EXPECT_EQ(countVideoFrames(pipeline.config.output_video), 6);
}
#endif

} // namespace
