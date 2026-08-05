// Both headers include KserveTypes.hpp from neuriplo-kserve-client, which is
// not fetched in a local-only build. Guarded like test_KserveEngine.cpp.
#ifdef NEURIPLO_INFER_WITH_KSERVE

#include "EncodedImage.hpp"
#include "KserveEnvelope.hpp"

#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

template <typename T> void append(std::vector<uint8_t> &bytes, T value) {
  const auto offset = bytes.size();
  bytes.resize(offset + sizeof(T));
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

kserve::InferOutput tensor(const std::string &name, const std::string &datatype,
                           std::vector<uint8_t> data) {
  kserve::InferOutput output;
  output.name = name;
  output.datatype = datatype;
  output.data = std::move(data);
  return output;
}

// Builds the padded detection envelope a conforming server returns: the arrays
// are always full length, only NUM_DETECTIONS says how much of them is real.
std::vector<kserve::InferOutput>
detectionEnvelope(int count, const std::vector<std::array<int32_t, 4>> &boxes,
                  const std::vector<float> &scores,
                  const std::vector<int32_t> &classes) {
  std::vector<uint8_t> count_bytes;
  append<int32_t>(count_bytes, count);

  std::vector<uint8_t> box_bytes;
  std::vector<uint8_t> score_bytes;
  std::vector<uint8_t> class_bytes;
  for (int i = 0; i < neuriplo_infer::kEnvelopeMaxDetections; ++i) {
    const bool valid = i < static_cast<int>(boxes.size());
    for (int axis = 0; axis < 4; ++axis) {
      append<int32_t>(
          box_bytes,
          valid ? boxes[static_cast<size_t>(i)][static_cast<size_t>(axis)] : 0);
    }
    append<float>(score_bytes, i < static_cast<int>(scores.size())
                                   ? scores[static_cast<size_t>(i)]
                                   : 0.0F);
    append<int32_t>(class_bytes, i < static_cast<int>(classes.size())
                                     ? classes[static_cast<size_t>(i)]
                                     : 0);
  }

  return {tensor("NUM_DETECTIONS", "INT32", count_bytes),
          tensor("BOXES", "INT32", box_bytes),
          tensor("SCORES", "FP32", score_bytes),
          tensor("CLASSES", "INT32", class_bytes)};
}

std::vector<uint8_t> fullMaskOffsets(const std::vector<int64_t> &lengths) {
  std::vector<uint8_t> bytes;
  int64_t offset = 0;
  append<int64_t>(bytes, offset);
  for (int i = 0; i < neuriplo_infer::kEnvelopeMaxDetections; ++i) {
    if (i < static_cast<int>(lengths.size())) {
      offset += lengths[static_cast<size_t>(i)];
    }
    append<int64_t>(bytes, offset);
  }
  return bytes;
}

} // namespace

TEST(KserveEnvelope, DecodesDetectionsAndIgnoresPadding) {
  auto outputs = detectionEnvelope(2, {{{10, 20, 30, 40}}, {{50, 60, 70, 80}}},
                                   {0.9F, 0.5F}, {3, 7});

  const auto results = neuriplo_infer::decodeDetectionEnvelope(outputs);
  // Only NUM_DETECTIONS rows are real; the other 98 are contractual padding.
  ASSERT_EQ(results.size(), 2u);

  const auto &first = std::get<neuriplo_tasks::Detection>(results[0]);
  EXPECT_FLOAT_EQ(first.bbox.x, 10.0F);
  EXPECT_FLOAT_EQ(first.bbox.height, 40.0F);
  EXPECT_FLOAT_EQ(first.class_confidence, 0.9F);
  EXPECT_FLOAT_EQ(first.class_id, 3.0F);

  const auto &second = std::get<neuriplo_tasks::Detection>(results[1]);
  EXPECT_FLOAT_EQ(second.bbox.x, 50.0F);
  EXPECT_FLOAT_EQ(second.class_id, 7.0F);
}

TEST(KserveEnvelope, DecodesEmptyFrameToNoDetections) {
  const auto outputs = detectionEnvelope(0, {}, {}, {});
  EXPECT_TRUE(neuriplo_infer::decodeDetectionEnvelope(outputs).empty());
}

TEST(KserveEnvelope, RejectsOutOfRangeDetectionCount) {
  auto outputs = detectionEnvelope(0, {}, {}, {});
  std::vector<uint8_t> bad;
  append<int32_t>(bad, 101);
  outputs[0] = tensor("NUM_DETECTIONS", "INT32", bad);
  EXPECT_THROW(neuriplo_infer::decodeDetectionEnvelope(outputs),
               std::runtime_error);
}

TEST(KserveEnvelope, RejectsMissingEnvelopeTensor) {
  auto outputs = detectionEnvelope(1, {{{1, 2, 3, 4}}}, {0.5F}, {0});
  outputs.pop_back(); // drop CLASSES
  EXPECT_THROW(neuriplo_infer::decodeDetectionEnvelope(outputs),
               std::runtime_error);
}

// Pins the placement regression: the envelope carries box-sized masks, and the
// decoder must expand them into a frame-sized image positioned at the
// detection's box. Returning a box-sized image instead made the renderer
// stretch one instance mask across the whole picture.
TEST(KserveEnvelope, PlacesMaskAtBoundingBoxOriginInFrameCoordinates) {
  auto outputs = detectionEnvelope(1, {{{4, 2, 2, 2}}}, {0.9F}, {0});
  outputs.push_back(tensor("MASK_OFFSETS", "INT64", fullMaskOffsets({4})));
  outputs.push_back(tensor("MASK_DATA", "UINT8", {255, 255, 255, 255}));

  const auto results = neuriplo_infer::decodeMaskEnvelope(outputs, 8, 6);
  ASSERT_EQ(results.size(), 1u);
  const auto &seg = std::get<neuriplo_tasks::InstanceSegmentation>(results[0]);

  // The mask image is frame-sized, not box-sized.
  ASSERT_FALSE(seg.mask.empty());
  EXPECT_EQ(seg.mask.cols(), 8);
  EXPECT_EQ(seg.mask.rows(), 6);

  // Set pixels appear only inside the box at (4,2) size 2x2.
  const auto *pixels = seg.mask.data();
  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 8; ++col) {
      const bool inside = row >= 2 && row < 4 && col >= 4 && col < 6;
      EXPECT_EQ(pixels[row * 8 + col] != 0, inside)
          << "pixel (" << col << "," << row << ")";
    }
  }
}

TEST(KserveEnvelope, DecodesPackedMasks) {
  auto outputs = detectionEnvelope(2, {{{0, 0, 2, 2}}, {{4, 4, 2, 2}}},
                                   {0.8F, 0.7F}, {1, 2});
  outputs.push_back(tensor("MASK_OFFSETS", "INT64", fullMaskOffsets({4, 4})));
  outputs.push_back(tensor("MASK_DATA", "UINT8", {1, 1, 0, 0, 0, 0, 1, 1}));

  const auto results = neuriplo_infer::decodeMaskEnvelope(outputs, 64, 64);
  ASSERT_EQ(results.size(), 2u);

  const auto &first =
      std::get<neuriplo_tasks::InstanceSegmentation>(results[0]);
  ASSERT_EQ(first.mask_data.size(), 4u);
  EXPECT_EQ(first.mask_data[0], 1);
  const auto &second =
      std::get<neuriplo_tasks::InstanceSegmentation>(results[1]);
  ASSERT_EQ(second.mask_data.size(), 4u);
  EXPECT_EQ(second.mask_data[3], 1);
}

// A server that shortens MASK_OFFSETS on a detection-free frame aborts video
// processing on the first empty frame. The client refuses the payload rather
// than reading past the end.
TEST(KserveEnvelope, RejectsTruncatedMaskOffsets) {
  auto outputs = detectionEnvelope(0, {}, {}, {});
  std::vector<uint8_t> short_offsets;
  append<int64_t>(short_offsets, 0);
  outputs.push_back(tensor("MASK_OFFSETS", "INT64", short_offsets));
  outputs.push_back(tensor("MASK_DATA", "UINT8", {}));

  EXPECT_THROW(neuriplo_infer::decodeMaskEnvelope(outputs), std::runtime_error);
}

TEST(KserveEnvelope, DecodesPolygonRingsAndHoles) {
  auto outputs = detectionEnvelope(1, {{{0, 0, 10, 10}}}, {0.9F}, {5});

  // One detection with two rings: an outer square and an inner square that
  // falls inside it, so the inner one is a hole.
  std::vector<uint8_t> instance_offsets;
  int64_t rings = 0;
  append<int64_t>(instance_offsets, rings);
  rings = 2;
  append<int64_t>(instance_offsets, rings);
  for (int i = 1; i < neuriplo_infer::kEnvelopeMaxDetections; ++i) {
    append<int64_t>(instance_offsets, rings);
  }

  std::vector<uint8_t> ring_offsets;
  append<int64_t>(ring_offsets, 0);
  append<int64_t>(ring_offsets, 4);
  append<int64_t>(ring_offsets, 8);

  std::vector<uint8_t> points;
  const int32_t outer[8] = {0, 0, 10, 0, 10, 10, 0, 10};
  const int32_t inner[8] = {3, 3, 6, 3, 6, 6, 3, 6};
  for (const auto value : outer) {
    append<int32_t>(points, value);
  }
  for (const auto value : inner) {
    append<int32_t>(points, value);
  }

  outputs.push_back(tensor("INSTANCE_RING_OFFSETS", "INT64", instance_offsets));
  outputs.push_back(tensor("RING_POINT_OFFSETS", "INT64", ring_offsets));
  outputs.push_back(tensor("POLYGON_POINTS", "INT32", points));

  const auto results = neuriplo_infer::decodePolygonEnvelope(outputs);
  ASSERT_EQ(results.size(), 1u);
  const auto &segmentation =
      std::get<neuriplo_tasks::InstanceSegmentation>(results[0]);
  ASSERT_EQ(segmentation.polygons.size(), 1u);
  EXPECT_EQ(segmentation.polygons[0].exterior.size(), 4u);
  ASSERT_EQ(segmentation.polygons[0].holes.size(), 1u);
  EXPECT_EQ(segmentation.polygons[0].holes[0].size(), 4u);
}

TEST(KserveEnvelope, RecognisesDecodedAndPassthroughModels) {
  kserve::ModelMetadata passthrough;
  passthrough.outputs.push_back({"output0", "FP32", {1, 84, 8400}});
  EXPECT_FALSE(neuriplo_infer::isDecodedEnvelope(passthrough));

  kserve::ModelMetadata decoded;
  decoded.outputs.push_back({"NUM_DETECTIONS", "INT32", {1}});
  decoded.outputs.push_back({"MASK_DATA", "UINT8", {-1}});
  EXPECT_TRUE(neuriplo_infer::isDecodedEnvelope(decoded));
  EXPECT_EQ(neuriplo_infer::envelopeVariantOf(decoded),
            neuriplo_infer::EnvelopeVariant::Mask);
}

TEST(EncodedImage, ValidatesEnsembleAgainstTaskModel) {
  kserve::ModelMetadata ensemble;
  ensemble.inputs.push_back({"IMAGE", "UINT8", {1, -1}});
  ensemble.outputs.push_back({"output0", "FP32", {1, 84, 8400}});

  kserve::ModelMetadata task_model;
  task_model.inputs.push_back({"images", "FP32", {1, 3, 640, 640}});
  task_model.outputs.push_back({"output0", "FP32", {1, 84, 8400}});

  EXPECT_NO_THROW(
      neuriplo_infer::validateEncodedImageModels(ensemble, task_model));

  // An ensemble wired to a different inner model is exactly what this check
  // exists to catch.
  kserve::ModelMetadata other_model = task_model;
  other_model.outputs[0].shape = {1, 84, 2100};
  EXPECT_THROW(
      neuriplo_infer::validateEncodedImageModels(ensemble, other_model),
      std::runtime_error);
}

TEST(EncodedImage, RejectsNonJpegPayload) {
  const std::vector<uint8_t> png = {0x89, 0x50, 0x4e, 0x47};
  EXPECT_THROW(neuriplo_infer::readJpegDimensions(png, "x.png"),
               std::runtime_error);
}

TEST(EncodedImage, ReadsJpegDimensionsFromStartOfFrame) {
  // SOI, then an SOF0 segment declaring 480x640, then EOI.
  std::vector<uint8_t> jpeg = {0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08};
  jpeg.push_back(0x01); // height high byte  (0x01e0 = 480)
  jpeg.push_back(0xe0);
  jpeg.push_back(0x02); // width high byte   (0x0280 = 640)
  jpeg.push_back(0x80);
  jpeg.resize(jpeg.size() + 8, 0x00);
  jpeg.push_back(0xff);
  jpeg.push_back(0xd9);

  const auto [width, height] =
      neuriplo_infer::readJpegDimensions(jpeg, "frame.jpg");
  EXPECT_EQ(width, 640);
  EXPECT_EQ(height, 480);
}

#endif // NEURIPLO_INFER_WITH_KSERVE
