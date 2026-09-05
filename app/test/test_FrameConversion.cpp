#include "FrameConversion.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using neuriplo_infer::toBgrMat;
using neuriplo_infer::toFrame;
using videocapture::Frame;
using videocapture::PixelFormat;

namespace {

// Fills a packed frame with a per-pixel pattern so a channel swap cannot pass
// by symmetry the way a solid color would.
void fillPacked(Frame &frame, const std::vector<std::uint8_t> &pixel) {
  for (int y = 0; y < frame.height(); ++y) {
    std::uint8_t *row = frame.data() + y * frame.rowStride();
    for (int x = 0; x < frame.width(); ++x) {
      for (std::size_t c = 0; c < pixel.size(); ++c) {
        row[x * pixel.size() + c] =
            static_cast<std::uint8_t>(pixel[c] + x + 2 * y);
      }
    }
  }
}

cv::Mat solidBgr(int width, int height, cv::Scalar color) {
  return cv::Mat(height, width, CV_8UC3, color);
}

} // namespace

TEST(FrameConversionTest, EmptyFrameConvertsToEmptyMat) {
  Frame frame;
  EXPECT_TRUE(toBgrMat(frame).empty());
}

TEST(FrameConversionTest, Bgr8AliasesFrameStorageWithoutCopying) {
  Frame frame(4, 3, PixelFormat::BGR8);
  fillPacked(frame, {10, 20, 30});

  cv::Mat mat = toBgrMat(frame);
  ASSERT_EQ(mat.type(), CV_8UC3);
  EXPECT_EQ(mat.cols, 4);
  EXPECT_EQ(mat.rows, 3);
  // The whole point of the packed-BGR8 path: same bytes, not a copy of them.
  EXPECT_EQ(mat.data, frame.data());

  // ...which is what makes the render overlay land in the frame's buffer.
  mat.at<cv::Vec3b>(1, 2) = cv::Vec3b(7, 8, 9);
  const std::uint8_t *pixel = frame.data() + frame.rowStride() + 2 * 3;
  EXPECT_EQ(pixel[0], 7);
  EXPECT_EQ(pixel[1], 8);
  EXPECT_EQ(pixel[2], 9);
}

TEST(FrameConversionTest, Rgb8IsReorderedToBgr) {
  Frame frame(4, 3, PixelFormat::RGB8);
  fillPacked(frame, {10, 20, 30});

  cv::Mat mat = toBgrMat(frame);
  ASSERT_EQ(mat.type(), CV_8UC3);
  EXPECT_NE(mat.data, frame.data()) << "a conversion must not alias the frame";

  const std::uint8_t *src = frame.data() + frame.rowStride() + 2 * 3;
  const cv::Vec3b converted = mat.at<cv::Vec3b>(1, 2);
  EXPECT_EQ(converted[0], src[2]);
  EXPECT_EQ(converted[1], src[1]);
  EXPECT_EQ(converted[2], src[0]);
}

TEST(FrameConversionTest, Gray8IsReplicatedAcrossBgrChannels) {
  Frame frame(4, 3, PixelFormat::Gray8);
  fillPacked(frame, {40});

  cv::Mat mat = toBgrMat(frame);
  ASSERT_EQ(mat.type(), CV_8UC3);
  const std::uint8_t gray = *(frame.data() + frame.rowStride() + 2);
  EXPECT_EQ(mat.at<cv::Vec3b>(1, 2), cv::Vec3b(gray, gray, gray));
}

TEST(FrameConversionTest, AlphaFormatsDropAlphaAndKeepBgrOrder) {
  Frame rgba(4, 3, PixelFormat::RGBA8);
  fillPacked(rgba, {10, 20, 30, 255});
  const std::uint8_t *rgbaPixel = rgba.data() + rgba.rowStride() + 2 * 4;
  const cv::Vec3b fromRgba = toBgrMat(rgba).at<cv::Vec3b>(1, 2);
  EXPECT_EQ(fromRgba, cv::Vec3b(rgbaPixel[2], rgbaPixel[1], rgbaPixel[0]));

  Frame bgra(4, 3, PixelFormat::BGRA8);
  fillPacked(bgra, {10, 20, 30, 255});
  const std::uint8_t *bgraPixel = bgra.data() + bgra.rowStride() + 2 * 4;
  const cv::Vec3b fromBgra = toBgrMat(bgra).at<cv::Vec3b>(1, 2);
  EXPECT_EQ(fromBgra, cv::Vec3b(bgraPixel[0], bgraPixel[1], bgraPixel[2]));
}

// The planar cases are the ones where a wrong plane layout would not fail --
// it would quietly decode a picture with the color torn out of place -- so
// they are checked against OpenCV's own I420 encoding of a known image.
TEST(FrameConversionTest, Yuv420pDecodesToTheOriginalColor) {
  const cv::Scalar color(32, 96, 200); // BGR
  cv::Mat i420;
  cv::cvtColor(solidBgr(8, 6, color), i420, cv::COLOR_BGR2YUV_I420);
  ASSERT_TRUE(i420.isContinuous());

  Frame frame(8, 6, PixelFormat::YUV420P);
  ASSERT_EQ(frame.storageSizeBytes(), i420.total());
  std::memcpy(frame.data(), i420.data, i420.total());

  cv::Mat bgr = toBgrMat(frame);
  ASSERT_EQ(bgr.size(), cv::Size(8, 6));
  const cv::Vec3b decoded = bgr.at<cv::Vec3b>(3, 4);
  EXPECT_NEAR(decoded[0], color[0], 3);
  EXPECT_NEAR(decoded[1], color[1], 3);
  EXPECT_NEAR(decoded[2], color[2], 3);
}

TEST(FrameConversionTest, Nv12DecodesToTheOriginalColor) {
  const cv::Scalar color(32, 96, 200); // BGR
  cv::Mat i420;
  cv::cvtColor(solidBgr(8, 6, color), i420, cv::COLOR_BGR2YUV_I420);

  // NV12 is I420 with the two chroma planes interleaved.
  Frame frame(8, 6, PixelFormat::NV12);
  const std::size_t lumaBytes = 8 * 6;
  const std::size_t chromaSamples = (8 / 2) * (6 / 2);
  ASSERT_EQ(frame.storageSizeBytes(), lumaBytes + 2 * chromaSamples);
  std::memcpy(frame.data(0), i420.data, lumaBytes);
  const std::uint8_t *u = i420.data + lumaBytes;
  const std::uint8_t *v = u + chromaSamples;
  std::uint8_t *chroma = frame.data(1);
  for (std::size_t i = 0; i < chromaSamples; ++i) {
    chroma[2 * i] = u[i];
    chroma[2 * i + 1] = v[i];
  }

  const cv::Vec3b decoded = toBgrMat(frame).at<cv::Vec3b>(3, 4);
  EXPECT_NEAR(decoded[0], color[0], 3);
  EXPECT_NEAR(decoded[1], color[1], 3);
  EXPECT_NEAR(decoded[2], color[2], 3);
}

TEST(FrameConversionTest, OddSized420FrameIsRejectedRatherThanMisread) {
  // Frame allows odd dimensions by rounding the chroma planes up, which breaks
  // the single-buffer layout OpenCV's 4:2:0 decoders require. Refusing beats
  // handing back a plausible-looking picture built from misaligned bytes.
  Frame frame(7, 6, PixelFormat::NV12);
  EXPECT_THROW(toBgrMat(frame), std::invalid_argument);
}

TEST(FrameConversionTest, ToFrameFromBgr8MatIsByteIdentical) {
  // A deterministic per-pixel pattern so a stride, size, or channel mix-up
  // cannot pass unnoticed the way a solid color might.
  cv::Mat mat(7, 8, CV_8UC3);
  ASSERT_TRUE(mat.isContinuous());
  for (int y = 0; y < mat.rows; ++y) {
    for (int x = 0; x < mat.cols; ++x) {
      mat.at<cv::Vec3b>(y, x) =
          cv::Vec3b(static_cast<std::uint8_t>(17 + x + 2 * y),
                    static_cast<std::uint8_t>(53 + x + 3 * y),
                    static_cast<std::uint8_t>(91 + x + 5 * y));
    }
  }

  const videocapture::Frame frame = toFrame(mat);
  ASSERT_FALSE(frame.empty());
  EXPECT_EQ(frame.width(), 8);
  EXPECT_EQ(frame.height(), 7);
  EXPECT_EQ(frame.format(), PixelFormat::BGR8);
  EXPECT_EQ(frame.rowStride(), static_cast<std::size_t>(8 * 3));
  // Tight-packed storage holds exactly the Mat's bytes.
  EXPECT_EQ(frame.rowStride() * frame.height(),
            static_cast<std::size_t>(mat.total()) * mat.elemSize());
  EXPECT_EQ(frame.storageSizeBytes(),
            static_cast<std::size_t>(mat.total()) * mat.elemSize());

  for (int y = 0; y < mat.rows; ++y) {
    EXPECT_EQ(0, std::memcmp(frame.data() + y * frame.rowStride(),
                             mat.data + y * mat.cols * 3,
                             static_cast<std::size_t>(mat.cols) * 3))
        << "row " << y << " differs from the source Mat";
  }
}

TEST(FrameConversionTest, ToFrameRejectsNonPackedBgr8) {
  cv::Mat gray(4, 3, CV_8UC1);
  try {
    toFrame(gray);
    FAIL() << "CV_8UC1 Mats must be rejected";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::strstr(error.what(), "bridged"), nullptr)
        << "rejection should name the toFrame bridging context: " << error.what();
  }

  cv::Mat floatBgr(4, 3, CV_32FC3);
  try {
    toFrame(floatBgr);
    FAIL() << "CV_32FC3 Mats must be rejected";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::strstr(error.what(), "bridged"), nullptr)
        << "rejection should name the toFrame bridging context: " << error.what();
  }
}

TEST(FrameConversionTest, ToFrameFromEmptyMatGivesEmptyFrame) {
  const videocapture::Frame frame = toFrame(cv::Mat{});
  EXPECT_TRUE(frame.empty());
  EXPECT_EQ(frame.storageSizeBytes(), 0u);
}
