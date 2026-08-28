#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

/**
 * Machine-readable diagnostics for one run.
 *
 * Consumers (the Neuriplo UI adapter, CI harnesses) need two things that
 * human-readable logs cannot supply reliably: where a failure happened, and
 * how long each stage took. Scraping glog text for either is brittle, so the
 * binary writes them to a versioned JSON document instead.
 *
 * Units and boundaries are part of the contract:
 *
 *   wall_time_ms          producer-measured lifetime of the run, in
 *                         milliseconds. It excludes process startup before
 *                         main() and is therefore always smaller than a
 *                         caller's own process wall time.
 *   stages_ms.<stage>     sum of every measured interval attributed to that
 *                         stage across the whole run, in milliseconds. A stage
 *                         that was never measured is absent, never zero.
 *   samples               number of sources processed to completion: one
 *                         still image, one video read to its end, one
 *                         optical-flow pair, one image-understanding request.
 *   frames                number of video frames processed, absent for a run
 *                         with no video source.
 *   throughput_per_second frames (or samples, when no frames were read)
 *                         divided by the accumulated inference seconds. It is
 *                         absent unless both boundaries were measured.
 *
 * Absent is not zero: a consumer must render nothing for a value it was not
 * given rather than report a measurement that was never taken.
 */

namespace neuriplo_infer {

/** Where a run was when it ended. Also used to attribute a failure. */
enum class RunStage : std::uint8_t {
  Configuration,
  ModelLoad,
  Source,
  Preprocess,
  Inference,
  Postprocess,
  Render,
  Unknown,
};

std::string_view toString(RunStage stage);

class RunReport {
public:
  /** Bumped only for a breaking change to the document below. */
  static constexpr int kSchemaVersion = 1;
  /** Written relative to the working directory, like the rendered output. */
  static constexpr const char *kDefaultPath = "data/output/run_report.json";

  RunReport();

  /** Marks what the run is doing, so a throw can be attributed to it. */
  void beginStage(RunStage stage);
  [[nodiscard]] RunStage currentStage() const { return stage_; }

  /** Accumulates a measured interval into a stage. */
  void addStageMs(RunStage stage, double milliseconds);

  void addSample();
  void addFrames(std::int64_t frames);

  /**
   * Stops and resumes timing accumulation, keeping stage attribution. Warmup
   * and benchmark iterations repeat inference to measure the engine, not to
   * produce a result: counting their time without counting their work would
   * inflate every stage total and collapse the reported throughput.
   */
  void suspendTiming();
  void resumeTiming();

  /**
   * Records a failure at an explicit stage, keeping the producer message. An
   * empty message means the producer said it on stderr and nowhere else, which
   * is reported as null rather than as an invented sentence.
   */
  void fail(RunStage stage, std::string message);
  [[nodiscard]] bool failed() const { return failure_.has_value(); }

  [[nodiscard]] nlohmann::json toJson() const;

private:
  [[nodiscard]] double elapsedMs() const;

  std::chrono::steady_clock::time_point started_at_;
  RunStage stage_{RunStage::Configuration};
  std::optional<double> model_load_ms_;
  std::optional<double> preprocess_ms_;
  std::optional<double> inference_ms_;
  std::optional<double> postprocess_ms_;
  std::optional<double> render_ms_;
  int timing_suspended_{0};
  std::int64_t samples_{0};
  std::int64_t frames_{0};
  bool saw_frames_{false};
  struct Failure {
    RunStage stage;
    std::string message;
  };
  std::optional<Failure> failure_;
};

/**
 * Writes the report, creating parent directories as needed, and returns whether
 * the complete document reached the disk. A short write leaves truncated JSON
 * that a consumer would reject, so the stream is flushed and checked rather
 * than only opened. Never throws: a diagnostics document that cannot be written
 * must not turn a successful run into a failed one, so a write error is logged
 * and reported through the return value.
 */
bool writeRunReport(const RunReport &report, const std::filesystem::path &path);

/** Convenience for a failure that happened before a pipeline existed. */
void writeConfigurationFailureReport(const std::string &message,
                                     const std::filesystem::path &path);

/**
 * Argument validation ends the process with `std::exit` rather than an
 * exception, so nothing unwinds back to main to describe it. Arming this
 * around the configuration phase writes a configuration-stage report if the
 * process leaves during it; the human message stays on stderr, where the
 * parser already put it.
 */
void armConfigurationExitReport(std::filesystem::path path);
void disarmConfigurationExitReport();

/**
 * Times a scope into a stage, and no-ops when no report is collecting. That
 * keeps instrumentation at the call sites to one line and keeps every existing
 * test, which builds pipelines without a report, working unchanged.
 */
/** Suspends timing for a scope; a null report makes it a no-op. */
class TimingSuspension {
public:
  explicit TimingSuspension(RunReport *report);
  ~TimingSuspension();

  TimingSuspension(const TimingSuspension &) = delete;
  TimingSuspension &operator=(const TimingSuspension &) = delete;

private:
  RunReport *report_;
};

class StageTimer {
public:
  StageTimer(RunReport *report, RunStage stage);
  ~StageTimer();

  StageTimer(const StageTimer &) = delete;
  StageTimer &operator=(const StageTimer &) = delete;

private:
  RunReport *report_;
  RunStage stage_;
  std::chrono::steady_clock::time_point started_at_;
};

} // namespace neuriplo_infer
