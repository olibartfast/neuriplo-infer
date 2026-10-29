#include "../validation/PredictionRecorder.hpp"

#include <cstring>
#include <gtest/gtest.h>

namespace {
using namespace neuriplo_tasks;
using neuriplo_infer::validation::PredictionRecorder;
using json = nlohmann::json;
namespace fs = std::filesystem;

class ModelValidationRecorder : public ::testing::Test {
protected:
  fs::path path_;
  void SetUp() override {
    path_ =
        fs::temp_directory_path() /
        ("neuriplo-model-validation-" +
         std::to_string(
             ::testing::UnitTest::GetInstance()->current_test_info()->line()));
    fs::remove_all(path_);
  }
  void TearDown() override { fs::remove_all(path_); }
};

TEST_F(ModelValidationRecorder, KeepsPredictionValuesAndOriginalCoordinates) {
  PredictionRecorder recorder(path_);
  recorder.predictions(Size(800, 600),
                       {Detection(BoundingBox(12, 23, 100, 200), 0.9f, 16)});
  const auto &sample = recorder.document["samples"][0];
  EXPECT_EQ(sample["image_size"], (json{800, 600}));
  EXPECT_EQ(sample["predictions"][0]["class_id"], 16.0f);
  EXPECT_EQ(sample["predictions"][0]["bbox"], (json{12, 23, 100, 200}));
  EXPECT_FLOAT_EQ(sample["predictions"][0]["score"].get<float>(), 0.9f);
  recorder.flush();
  std::ifstream stream(path_ / "predictions.json");
  EXPECT_EQ(json::parse(stream), recorder.document);
}

TEST_F(ModelValidationRecorder, PreservesRawFloatMapGeometryAndBytes) {
  PredictionRecorder recorder(path_);
  Image image = Image::zeros(2, 1, 1, PixelType::Float32);
  image.data<float>()[0] = 1.25f;
  image.data<float>()[1] = -2.5f;
  const auto descriptor = recorder.matrix(fromImage(std::move(image)));
  EXPECT_EQ(descriptor["shape"], (json{1, 2, 1}));
  EXPECT_EQ(descriptor["dtype"], "FP32");
  std::ifstream stream(path_ / descriptor["file"].get<std::string>(),
                       std::ios::binary);
  float values[2]{};
  stream.read(reinterpret_cast<char *>(values), sizeof(values));
  ASSERT_TRUE(stream);
  EXPECT_FLOAT_EQ(values[0], 1.25f);
  EXPECT_FLOAT_EQ(values[1], -2.5f);
}

TEST_F(ModelValidationRecorder, KeepsInt64TensorTypeWithoutFloatConversion) {
  PredictionRecorder recorder(path_);
  Tensor tensor;
  tensor.shape = {1, 2};
  tensor.data = {int64_t{9007199254740993LL}, int64_t{7}};
  const auto descriptor = recorder.tensor(tensor);
  EXPECT_EQ(descriptor["dtype"], "INT64");
  EXPECT_EQ(descriptor["shape"], (json{1, 2}));
  std::ifstream stream(path_ / descriptor["file"].get<std::string>(),
                       std::ios::binary);
  int64_t values[2]{};
  stream.read(reinterpret_cast<char *>(values), sizeof(values));
  ASSERT_TRUE(stream);
  EXPECT_EQ(values[0], 9007199254740993LL);
  EXPECT_EQ(values[1], 7);
}

TEST_F(ModelValidationRecorder, HandlesEveryResultVariant) {
  PredictionRecorder recorder(path_);
  const std::vector<Result> values = {
      Classification{},           Detection{},       OpenVocabDetection{},
      InstanceSegmentation{},     OpticalFlow{},     VideoClassification{},
      PoseEstimation{},           DepthEstimation{}, GaussianSplatting{},
      ImageUnderstanding{"a dog"}};
  for (const auto &value : values) {
    EXPECT_TRUE(recorder.result(value).contains("kind"));
  }
}

TEST_F(ModelValidationRecorder, RejectsMixedTensorTypesAndExistingEvidence) {
  PredictionRecorder recorder(path_);
  EXPECT_THROW(recorder.tensor(Tensor({1.0f, int64_t{2}}, {1, 2})),
               std::runtime_error);
  recorder.flush();
  EXPECT_THROW(PredictionRecorder repeated(path_), std::runtime_error);
}
} // namespace
