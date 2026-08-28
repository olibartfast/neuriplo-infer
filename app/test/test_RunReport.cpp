#include "RunReport.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <thread>

namespace {

using json = nlohmann::json;
using neuriplo_infer::RunReport;
using neuriplo_infer::RunStage;
using neuriplo_infer::StageTimer;

json readReport(const std::filesystem::path &path) {
  std::ifstream stream(path);
  EXPECT_TRUE(stream.is_open()) << "no report written at " << path;
  return json::parse(stream);
}

// A run report is only useful to a machine, so every test here reads the
// document rather than the collector's C++ surface.
class RunReportFile : public ::testing::Test {
protected:
  void SetUp() override {
    directory_ =
        std::filesystem::temp_directory_path() /
        ("neuriplo-run-report-" +
         std::to_string(
             ::testing::UnitTest::GetInstance()->current_test_info()->line()));
    std::filesystem::remove_all(directory_);
    path_ = directory_ / "run_report.json";
  }

  void TearDown() override { std::filesystem::remove_all(directory_); }

  std::filesystem::path directory_;
  std::filesystem::path path_;
};

TEST_F(RunReportFile, SuccessfulRunReportsNoFailure) {
  RunReport report;
  report.beginStage(RunStage::Render);
  report.addSample();

  neuriplo_infer::writeRunReport(report, path_);
  const json document = readReport(path_);

  EXPECT_EQ(document.at("schema_version"), RunReport::kSchemaVersion);
  EXPECT_EQ(document.at("status"), "success");
  EXPECT_TRUE(document.at("error").is_null());
  EXPECT_EQ(document.at("metrics").at("samples"), 1);
  EXPECT_GE(document.at("metrics").at("wall_time_ms").get<double>(), 0.0);
}

TEST_F(RunReportFile, UnmeasuredStagesAreNullRatherThanZero) {
  RunReport report;
  report.addStageMs(RunStage::Inference, 12.5);

  neuriplo_infer::writeRunReport(report, path_);
  const json stages = readReport(path_).at("metrics").at("stages_ms");

  EXPECT_DOUBLE_EQ(stages.at("inference").get<double>(), 12.5);
  // A stage nobody measured must not be reported as having taken no time.
  EXPECT_TRUE(stages.at("model_load").is_null());
  EXPECT_TRUE(stages.at("preprocess").is_null());
  EXPECT_TRUE(stages.at("postprocess").is_null());
  EXPECT_TRUE(stages.at("render").is_null());
}

TEST_F(RunReportFile, StageTimesAccumulateAcrossFrames) {
  RunReport report;
  report.addStageMs(RunStage::Inference, 10.0);
  report.addStageMs(RunStage::Inference, 30.0);

  neuriplo_infer::writeRunReport(report, path_);
  const json metrics = readReport(path_).at("metrics");

  // The contract says a stage value is the sum over the run.
  EXPECT_DOUBLE_EQ(metrics.at("stages_ms").at("inference").get<double>(), 40.0);
}

TEST_F(RunReportFile, ThroughputNeedsBothBoundaries) {
  RunReport counted_only;
  counted_only.addFrames(8);
  neuriplo_infer::writeRunReport(counted_only, path_);
  EXPECT_TRUE(
      readReport(path_).at("metrics").at("throughput_per_second").is_null());

  RunReport timed_only;
  timed_only.addStageMs(RunStage::Inference, 100.0);
  neuriplo_infer::writeRunReport(timed_only, path_);
  EXPECT_TRUE(
      readReport(path_).at("metrics").at("throughput_per_second").is_null());

  RunReport both;
  both.addFrames(8);
  both.addStageMs(RunStage::Inference, 100.0);
  neuriplo_infer::writeRunReport(both, path_);
  const json metrics = readReport(path_).at("metrics");
  EXPECT_EQ(metrics.at("frames"), 8);
  EXPECT_DOUBLE_EQ(metrics.at("throughput_per_second").get<double>(), 80.0);
}

TEST_F(RunReportFile, StillImageRunReportsNoFrameCount) {
  RunReport report;
  report.addSample();
  report.addStageMs(RunStage::Inference, 250.0);

  neuriplo_infer::writeRunReport(report, path_);
  const json metrics = readReport(path_).at("metrics");

  EXPECT_TRUE(metrics.at("frames").is_null());
  EXPECT_EQ(metrics.at("samples"), 1);
  EXPECT_DOUBLE_EQ(metrics.at("throughput_per_second").get<double>(), 4.0);
}

TEST_F(RunReportFile, FailureIsAttributedToTheStageItReached) {
  RunReport report;
  report.beginStage(RunStage::ModelLoad);
  report.fail(report.currentStage(), "Can't setup an inference engine");

  neuriplo_infer::writeRunReport(report, path_);
  const json document = readReport(path_);

  EXPECT_EQ(document.at("status"), "failed");
  EXPECT_EQ(document.at("stage"), "model_load");
  EXPECT_EQ(document.at("error").at("stage"), "model_load");
  EXPECT_EQ(document.at("error").at("message"),
            "Can't setup an inference engine");
}

TEST_F(RunReportFile, FirstFailureWins) {
  RunReport report;
  report.fail(RunStage::Inference, "the real cause");
  report.fail(RunStage::Render, "a later symptom");

  neuriplo_infer::writeRunReport(report, path_);
  const json error = readReport(path_).at("error");

  EXPECT_EQ(error.at("stage"), "inference");
  EXPECT_EQ(error.at("message"), "the real cause");
}

TEST_F(RunReportFile, MessagelessFailureReportsNullRatherThanAnInventedOne) {
  neuriplo_infer::writeConfigurationFailureReport({}, path_);
  const json document = readReport(path_);

  EXPECT_EQ(document.at("status"), "failed");
  EXPECT_EQ(document.at("error").at("stage"), "configuration");
  EXPECT_TRUE(document.at("error").at("message").is_null());
}

TEST_F(RunReportFile, EveryStageHasAStableName) {
  EXPECT_EQ(neuriplo_infer::toString(RunStage::Configuration), "configuration");
  EXPECT_EQ(neuriplo_infer::toString(RunStage::ModelLoad), "model_load");
  EXPECT_EQ(neuriplo_infer::toString(RunStage::Source), "source");
  EXPECT_EQ(neuriplo_infer::toString(RunStage::Preprocess), "preprocess");
  EXPECT_EQ(neuriplo_infer::toString(RunStage::Inference), "inference");
  EXPECT_EQ(neuriplo_infer::toString(RunStage::Postprocess), "postprocess");
  EXPECT_EQ(neuriplo_infer::toString(RunStage::Render), "render");
  EXPECT_EQ(neuriplo_infer::toString(RunStage::Unknown), "unknown");
}

TEST_F(RunReportFile, StageTimerMeasuresItsScopeAndMarksTheStage) {
  RunReport report;
  {
    StageTimer timer(&report, RunStage::Preprocess);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(report.currentStage(), RunStage::Preprocess);
  }

  neuriplo_infer::writeRunReport(report, path_);
  const json stages = readReport(path_).at("metrics").at("stages_ms");
  EXPECT_GE(stages.at("preprocess").get<double>(), 4.0);
}

TEST_F(RunReportFile, StageTimerWithoutACollectorIsANoOp) {
  // Library users and existing tests build pipelines with no report; the
  // instrumentation must not require one.
  EXPECT_NO_FATAL_FAILURE({ StageTimer timer(nullptr, RunStage::Inference); });
}

TEST_F(RunReportFile, WritingCreatesMissingParentDirectories) {
  RunReport report;
  const auto nested = directory_ / "data" / "output" / "run_report.json";

  neuriplo_infer::writeRunReport(report, nested);

  EXPECT_TRUE(std::filesystem::exists(nested));
}

TEST_F(RunReportFile, AnUnwritableReportNeverThrows) {
  RunReport report;
  // A directory where the file should be: writing must fail quietly, because
  // diagnostics may not change the outcome they describe.
  std::filesystem::create_directories(path_);

  EXPECT_NO_THROW(neuriplo_infer::writeRunReport(report, path_));
}

} // namespace
