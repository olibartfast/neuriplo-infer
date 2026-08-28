#include "RunReport.hpp"

#include <glog/logging.h>

#include <cstdlib>
#include <fstream>
#include <utility>

namespace neuriplo_infer {
namespace {

using json = nlohmann::json;

/** Absent stays absent: an unmeasured value is null, never 0. */
void assignOptional(json &target, const char *key,
                    const std::optional<double> &value) {
  target[key] = value.has_value() ? json(*value) : json(nullptr);
}

void accumulate(std::optional<double> &target, double milliseconds) {
  target = target.value_or(0.0) + milliseconds;
}

} // namespace

std::string_view toString(RunStage stage) {
  switch (stage) {
  case RunStage::Configuration:
    return "configuration";
  case RunStage::ModelLoad:
    return "model_load";
  case RunStage::Source:
    return "source";
  case RunStage::Preprocess:
    return "preprocess";
  case RunStage::Inference:
    return "inference";
  case RunStage::Postprocess:
    return "postprocess";
  case RunStage::Render:
    return "render";
  case RunStage::Unknown:
    break;
  }
  return "unknown";
}

RunReport::RunReport() : started_at_(std::chrono::steady_clock::now()) {}

void RunReport::beginStage(RunStage stage) { stage_ = stage; }

void RunReport::addStageMs(RunStage stage, double milliseconds) {
  if (milliseconds < 0.0) {
    return;
  }
  switch (stage) {
  case RunStage::ModelLoad:
    accumulate(model_load_ms_, milliseconds);
    return;
  case RunStage::Preprocess:
    accumulate(preprocess_ms_, milliseconds);
    return;
  case RunStage::Inference:
    accumulate(inference_ms_, milliseconds);
    return;
  case RunStage::Postprocess:
    accumulate(postprocess_ms_, milliseconds);
    return;
  case RunStage::Render:
    accumulate(render_ms_, milliseconds);
    return;
  case RunStage::Configuration:
  case RunStage::Source:
  case RunStage::Unknown:
    // Not timed stages: attribution only.
    return;
  }
}

void RunReport::addSample() { ++samples_; }

void RunReport::addFrames(std::int64_t frames) {
  if (frames <= 0) {
    return;
  }
  saw_frames_ = true;
  frames_ += frames;
}

void RunReport::fail(RunStage stage, std::string message) {
  if (failure_.has_value()) {
    return; // The first failure is the one that ended the run.
  }
  failure_ = Failure{stage, std::move(message)};
}

namespace {
std::filesystem::path g_configuration_report_path;
bool g_configuration_report_armed = false;
bool g_configuration_hook_registered = false;

void writeConfigurationReportAtExit() {
  if (!g_configuration_report_armed) {
    return;
  }
  g_configuration_report_armed = false;
  writeConfigurationFailureReport({}, g_configuration_report_path);
}
} // namespace

void armConfigurationExitReport(std::filesystem::path path) {
  g_configuration_report_path = std::move(path);
  g_configuration_report_armed = true;
  if (!g_configuration_hook_registered) {
    g_configuration_hook_registered =
        std::atexit(writeConfigurationReportAtExit) == 0;
  }
}

void disarmConfigurationExitReport() { g_configuration_report_armed = false; }

double RunReport::elapsedMs() const {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now - started_at_).count();
}

json RunReport::toJson() const {
  json stages = json::object();
  assignOptional(stages, "model_load", model_load_ms_);
  assignOptional(stages, "preprocess", preprocess_ms_);
  assignOptional(stages, "inference", inference_ms_);
  assignOptional(stages, "postprocess", postprocess_ms_);
  assignOptional(stages, "render", render_ms_);

  // Throughput needs both boundaries measured, so it stays absent unless a
  // processed count and the inference time it belongs to are both known.
  const std::int64_t processed = saw_frames_ ? frames_ : samples_;
  json throughput = nullptr;
  if (processed > 0 && inference_ms_.has_value() && *inference_ms_ > 0.0) {
    throughput = static_cast<double>(processed) / (*inference_ms_ / 1000.0);
  }

  json metrics = {
      {"wall_time_ms", elapsedMs()},
      {"samples", samples_},
      {"frames", saw_frames_ ? json(frames_) : json(nullptr)},
      {"throughput_per_second", throughput},
      {"stages_ms", std::move(stages)},
  };

  json document = {
      {"schema_version", kSchemaVersion},
      {"status", failure_.has_value() ? "failed" : "success"},
      {"stage",
       std::string(toString(failure_.has_value() ? failure_->stage : stage_))},
      {"metrics", std::move(metrics)},
  };

  document["error"] =
      failure_.has_value()
          ? json{{"stage", std::string(toString(failure_->stage))},
                 {"message", failure_->message.empty()
                                 ? json(nullptr)
                                 : json(failure_->message)}}
          : json(nullptr);

  return document;
}

void writeRunReport(const RunReport &report,
                    const std::filesystem::path &path) {
  try {
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path, std::ios::trunc);
    if (!stream.is_open()) {
      LOG(WARNING) << "Could not write the run report to " << path;
      return;
    }
    stream << report.toJson().dump(2) << '\n';
  } catch (const std::exception &e) {
    // Diagnostics must never change the outcome they describe.
    LOG(WARNING) << "Could not write the run report: " << e.what();
  }
}

void writeConfigurationFailureReport(const std::string &message,
                                     const std::filesystem::path &path) {
  RunReport report;
  report.fail(RunStage::Configuration, message);
  writeRunReport(report, path);
}

StageTimer::StageTimer(RunReport *report, RunStage stage)
    : report_(report), stage_(stage),
      started_at_(std::chrono::steady_clock::now()) {
  if (report_ != nullptr) {
    report_->beginStage(stage);
  }
}

StageTimer::~StageTimer() {
  if (report_ == nullptr) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  report_->addStageMs(
      stage_,
      std::chrono::duration<double, std::milli>(now - started_at_).count());
}

} // namespace neuriplo_infer
