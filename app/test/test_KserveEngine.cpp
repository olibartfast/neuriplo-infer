#ifdef NEURIPLO_INFER_WITH_KSERVE

#include "KserveEngine.hpp"
#include "KserveTypes.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace {

template <typename Metadata>
void setMetadataPlatform(Metadata &metadata, const std::string &platform) {
  if constexpr (requires { metadata.platform; }) {
    metadata.platform = platform;
  }
}

template <typename Metadata> constexpr bool hasMetadataPlatform() {
  return requires(Metadata metadata) { metadata.platform; };
}

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
    md.inputs = inputs_;
    md.outputs.push_back({"output", "FP32", {1, 1}});
    setMetadataPlatform(md, platform_);
    return md;
  }

  void setPlatform(std::string platform) { platform_ = std::move(platform); }

  void setInputs(std::vector<kserve::TensorSpec> inputs) {
    inputs_ = std::move(inputs);
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
  std::string platform_;
  std::vector<kserve::TensorSpec> inputs_{{"input", "FP32", {1, 1}}};
};

std::vector<std::vector<uint8_t>> oneFloatInput() {
  std::vector<uint8_t> bytes(sizeof(float));
  const float value = 0.5F;
  std::memcpy(bytes.data(), &value, sizeof(float));
  return {bytes};
}

std::vector<uint8_t> floatBytes(size_t count) {
  return std::vector<uint8_t>(count * sizeof(float));
}

} // namespace

TEST(KserveEngine, CarriesAdvertisedInputDatatypesIntoMetadata) {
  auto client = std::make_unique<FakeClient>();
  client->setInputs({{"a", "INT8", {1, 4}},
                     {"b", "BOOL", {1, 4}},
                     {"c", "UINT8", {1, 4}},
                     {"d", "INT64", {1, 2}},
                     {"e", "FP16", {1, 4}}});
  KserveEngine engine(std::move(client));

  const auto inputs = engine.get_inference_metadata().getInputs();

  ASSERT_EQ(inputs.size(), 5U);
  EXPECT_EQ(inputs[0].datatype, TensorDataType::Int8);
  EXPECT_EQ(inputs[1].datatype, TensorDataType::Bool);
  EXPECT_EQ(inputs[2].datatype, TensorDataType::UInt8);
  EXPECT_EQ(inputs[3].datatype, TensorDataType::Int64);
  // FP16 has no metadata member; callers that need it read rawMetadata().
  EXPECT_EQ(inputs[4].datatype, TensorDataType::Float32);
}

TEST(KserveEngine, RefusesBytesThatDoNotFitTheAdvertisedDatatype) {
  for (const std::string datatype : {"INT8", "BOOL", "UINT8", "FP16"}) {
    auto client = std::make_unique<FakeClient>();
    const FakeClient *fake = client.get();
    client->setInputs({{"images", datatype, {1, 4}}});
    KserveEngine engine(std::move(client));

    // Four Float32 elements are 16 bytes, which none of these datatypes holds.
    EXPECT_THROW(engine.get_infer_results({floatBytes(4)}), std::runtime_error)
        << datatype;
    EXPECT_EQ(fake->inferCalls(), 0) << "nothing may be sent: " << datatype;
  }
}

TEST(KserveEngine, NamesTheInputAndDatatypeWhenRefusing) {
  auto client = std::make_unique<FakeClient>();
  client->setInputs({{"pixel_values", "INT8", {1, 3, 2, 2}}});
  KserveEngine engine(std::move(client));

  try {
    engine.get_infer_results({floatBytes(12)});
    FAIL() << "Float32 bytes under INT8 must be refused";
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("'pixel_values'"), std::string::npos) << message;
    EXPECT_NE(message.find("INT8"), std::string::npos) << message;
  }
}

TEST(KserveEngine, SendsBytesThatFitTheAdvertisedDatatype) {
  auto client = std::make_unique<FakeClient>();
  const FakeClient *fake = client.get();
  client->setInputs({{"images", "UINT8", {1, 4}},
                     {"orig_target_sizes", "INT64", {1, 2}},
                     {"encoded", "UINT8", {-1}},
                     {"dynamic_batch", "FP32", {-1, 4}}});
  KserveEngine engine(std::move(client));

  engine.get_infer_results({std::vector<uint8_t>(4), std::vector<uint8_t>(16),
                            std::vector<uint8_t>(1234), floatBytes(8)});

  EXPECT_EQ(fake->inferCalls(), 1);
}

TEST(KserveEngine, RefusesAShapeWhoseByteCountOverflows) {
  auto client = std::make_unique<FakeClient>();
  const FakeClient *fake = client.get();
  // The static dimensions overflow size_t when multiplied by the FP32 width,
  // so the guard must reject the metadata instead of wrapping to a small value.
  client->setInputs(
      {{"images", "FP32", {1, 1, std::numeric_limits<int64_t>::max(), 2}}});
  KserveEngine engine(std::move(client));

  EXPECT_THROW(engine.get_infer_results({floatBytes(1)}), std::runtime_error);
  EXPECT_EQ(fake->inferCalls(), 0);
}

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

TEST(KserveEngine, ServingPlatformEmptyBeforeMetadataFetch) {
  KserveEngine engine(std::make_unique<FakeClient>());
  EXPECT_TRUE(engine.servingPlatform().empty());
}

TEST(KserveEngine, ExposesServingPlatformFromMetadata) {
  auto client = std::make_unique<FakeClient>();
  client->setPlatform("tensorrt_plan");
  KserveEngine engine(std::move(client));

  // Fetching metadata is what populates the served platform.
  engine.get_inference_metadata();

  if constexpr (hasMetadataPlatform<kserve::ModelMetadata>()) {
    EXPECT_EQ(engine.servingPlatform(), "tensorrt_plan");
  } else {
    EXPECT_TRUE(engine.servingPlatform().empty());
  }
}

#endif // NEURIPLO_INFER_WITH_KSERVE
