#ifdef NEURIPLO_INFER_WITH_KSERVE

#include "KserveEngine.hpp"
#include "KserveTypes.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

namespace {

// Minimal in-memory KServe protocol client so KserveEngine can be exercised
// without a real server. Reports a single FP32 input/output and echoes a fixed
// payload back; infer() optionally sleeps so latency is measurably non-zero.
class FakeClient : public kserve::IClient {
public:
  explicit FakeClient(
      std::chrono::milliseconds infer_delay = std::chrono::milliseconds(0))
      : infer_delay_(infer_delay) {}

  kserve::ModelMetadata modelMetadata() override {
    kserve::ModelMetadata md;
    md.inputs.push_back({"input", "FP32", {1, 1}});
    md.outputs.push_back({"output", "FP32", {1, 1}});
    return md;
  }

  std::vector<kserve::InferOutput>
  infer(const std::vector<kserve::InferInput> &inputs) override {
    (void)inputs;
    ++infer_calls_;
    if (infer_delay_.count() > 0) {
      std::this_thread::sleep_for(infer_delay_);
    }
    kserve::InferOutput out;
    out.name = "output";
    out.datatype = "FP32";
    out.shape = {1, 1};
    const float value = 1.0F;
    out.data.resize(sizeof(float));
    std::memcpy(out.data.data(), &value, sizeof(float));
    return {out};
  }

  bool serverLive() override { return true; }
  bool serverReady() override { return true; }
  bool modelReady() override { return true; }

  int inferCalls() const { return infer_calls_; }

private:
  std::chrono::milliseconds infer_delay_;
  int infer_calls_{0};
};

std::vector<std::vector<uint8_t>> oneFloatInput() {
  std::vector<uint8_t> bytes(sizeof(float));
  const float value = 0.5F;
  std::memcpy(bytes.data(), &value, sizeof(float));
  return {bytes};
}

} // namespace

TEST(KserveEngine, LatencyStartsAtZero) {
  KserveEngine engine(std::make_unique<FakeClient>());
  EXPECT_EQ(engine.inferenceCount(), 0U);
  EXPECT_DOUBLE_EQ(engine.lastInferenceLatencyMs(), 0.0);
  EXPECT_DOUBLE_EQ(engine.averageInferenceLatencyMs(), 0.0);
}

TEST(KserveEngine, TracksPerRequestLatency) {
  KserveEngine engine(
      std::make_unique<FakeClient>(std::chrono::milliseconds(5)));

  engine.get_infer_results(oneFloatInput());

  EXPECT_EQ(engine.inferenceCount(), 1U);
  EXPECT_GT(engine.lastInferenceLatencyMs(), 0.0);
  // Single request: average equals the last sample.
  EXPECT_DOUBLE_EQ(engine.averageInferenceLatencyMs(),
                   engine.lastInferenceLatencyMs());
}

TEST(KserveEngine, AggregatesAcrossRequests) {
  KserveEngine engine(std::make_unique<FakeClient>());

  for (int i = 0; i < 3; ++i) {
    engine.get_infer_results(oneFloatInput());
  }

  EXPECT_EQ(engine.inferenceCount(), 3U);
  EXPECT_GE(engine.lastInferenceLatencyMs(), 0.0);
  EXPECT_GE(engine.averageInferenceLatencyMs(), 0.0);
}

#endif // NEURIPLO_INFER_WITH_KSERVE
