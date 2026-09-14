#include "ModelInputTypes.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

neuriplo_tasks::ModelInfo modelWithInputs(
    const std::vector<std::pair<std::string, std::vector<int64_t>>> &inputs) {
  neuriplo_tasks::ModelInfo info;
  for (const auto &[name, shape] : inputs) {
    info.addInput(name, shape, 1);
  }
  return info;
}

std::string rejection(neuriplo_tasks::ModelInfo &info,
                      const std::vector<std::string> &datatypes) {
  try {
    neuriplo_infer::applyInputDatatypes(info, datatypes, false);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return {};
}

} // namespace

TEST(ModelInputTypes, TagsEveryBackendDatatype) {
  EXPECT_EQ(neuriplo_infer::datatypeTag(TensorDataType::Float32), "FP32");
  EXPECT_EQ(neuriplo_infer::datatypeTag(TensorDataType::Int32), "INT32");
  EXPECT_EQ(neuriplo_infer::datatypeTag(TensorDataType::Int64), "INT64");
  EXPECT_EQ(neuriplo_infer::datatypeTag(TensorDataType::UInt8), "UINT8");
  EXPECT_EQ(neuriplo_infer::datatypeTag(TensorDataType::Int8), "INT8");
  EXPECT_EQ(neuriplo_infer::datatypeTag(TensorDataType::Bool), "BOOL");
}

TEST(ModelInputTypes, Fp32ImageInputStaysFloat32) {
  auto info = modelWithInputs({{"images", {1, 3, 640, 640}}});
  neuriplo_infer::applyInputDatatypes(info, {"FP32"}, false);
  EXPECT_EQ(info.input_types[0], neuriplo_tasks::PixelType::Float32);
}

TEST(ModelInputTypes, Uint8ImageInputBecomesUInt8) {
  auto info = modelWithInputs({{"images", {1, 3, 640, 640}}});
  neuriplo_infer::applyInputDatatypes(info, {"UINT8"}, false);
  EXPECT_EQ(info.input_types[0], neuriplo_tasks::PixelType::UInt8);
}

TEST(ModelInputTypes, RejectsImageInputsPreprocessingCannotProduce) {
  for (const std::string datatype : {"INT8", "BOOL", "FP16", "INT64"}) {
    auto info = modelWithInputs({{"pixel_values", {1, 3, 32, 32}}});
    const auto message = rejection(info, {datatype});
    EXPECT_NE(message.find("'pixel_values'"), std::string::npos) << message;
    EXPECT_NE(message.find(datatype), std::string::npos) << message;
  }
}

TEST(ModelInputTypes, LeavesNonImageInputsToTheTask) {
  auto info = modelWithInputs(
      {{"images", {1, 3, 640, 640}}, {"orig_target_sizes", {1, 2}}});
  neuriplo_infer::applyInputDatatypes(info, {"FP32", "INT64"}, false);
  EXPECT_EQ(info.input_types[1], neuriplo_tasks::PixelType::Float32);
}

TEST(ModelInputTypes, ServerSidePreprocessingIsNotChecked) {
  auto info = modelWithInputs({{"images", {1, 3, 640, 640}}});
  EXPECT_NO_THROW(neuriplo_infer::applyInputDatatypes(info, {"INT8"}, true));
  EXPECT_EQ(info.input_types[0], neuriplo_tasks::PixelType::Float32);
}
