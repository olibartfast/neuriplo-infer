#include "FrameConversion.hpp"
#include "VideoCaptureFactory.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#ifdef VIDEOCAPTURE_WITH_WRITER
#include "VideoWriterConfig.hpp"
#include "VideoWriterFactory.hpp"
#endif

using neuriplo_infer::toBgrImage;
using neuriplo_infer::toFrame;
using neuriplo_tasks::vision::Image;
using neuriplo_tasks::vision::PixelType;
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

Image solidImage(int width, int height, std::uint8_t b, std::uint8_t g,
                 std::uint8_t r) {
  Image image = Image::uninit(width, height, 3, PixelType::UInt8);
  for (int y = 0; y < height; ++y) {
    std::uint8_t *row = image.ptr<std::uint8_t>(y);
    for (int x = 0; x < width; ++x) {
      row[3 * x] = b;
      row[3 * x + 1] = g;
      row[3 * x + 2] = r;
    }
  }
  return image;
}

} // namespace

TEST(FrameConversionTest, EmptyFrameConvertsToEmptyImage) {
  Frame frame;
  EXPECT_TRUE(toBgrImage(frame).empty());
}

TEST(FrameConversionTest, Bgr8CopiesFrameStorage) {
  Frame frame(4, 3, PixelFormat::BGR8);
  fillPacked(frame, {10, 20, 30});

  Image image = toBgrImage(frame);
  ASSERT_EQ(image.width(), 4);
  ASSERT_EQ(image.height(), 3);
  ASSERT_EQ(image.channels(), 3);
  // The Image backend always owns its pixels (no aliasing constructor).
  EXPECT_NE(image.raw(), frame.data());
  EXPECT_EQ(0, std::memcmp(image.raw(), frame.data(), 4 * 3 * 3));

  // Mutating the image must not alias the frame's buffer.
  image.ptr<std::uint8_t>(1)[2 * 3] = 7;
  image.ptr<std::uint8_t>(1)[2 * 3 + 1] = 8;
  image.ptr<std::uint8_t>(1)[2 * 3 + 2] = 9;
  const std::uint8_t *pixel = frame.data() + frame.rowStride() + 2 * 3;
  EXPECT_NE(pixel[0], 7);
  EXPECT_EQ(pixel[0], static_cast<std::uint8_t>(10 + 2 + 2));
}

TEST(FrameConversionTest, Rgb8IsReorderedToBgr) {
  Frame frame(4, 3, PixelFormat::RGB8);
  fillPacked(frame, {10, 20, 30});

  Image image = toBgrImage(frame);
  ASSERT_EQ(image.channels(), 3);

  const std::uint8_t *src = frame.data() + frame.rowStride() + 2 * 3;
  const std::uint8_t *converted = image.ptr<std::uint8_t>(1) + 2 * 3;
  EXPECT_EQ(converted[0], src[2]);
  EXPECT_EQ(converted[1], src[1]);
  EXPECT_EQ(converted[2], src[0]);
}

TEST(FrameConversionTest, Gray8IsReplicatedAcrossBgrChannels) {
  Frame frame(4, 3, PixelFormat::Gray8);
  fillPacked(frame, {40});

  Image image = toBgrImage(frame);
  const std::uint8_t gray = *(frame.data() + frame.rowStride() + 2);
  const std::uint8_t *pixel = image.ptr<std::uint8_t>(1) + 2 * 3;
  EXPECT_EQ(pixel[0], gray);
  EXPECT_EQ(pixel[1], gray);
  EXPECT_EQ(pixel[2], gray);
}

TEST(FrameConversionTest, AlphaFormatsDropAlphaAndKeepBgrOrder) {
  Frame rgba(4, 3, PixelFormat::RGBA8);
  fillPacked(rgba, {10, 20, 30, 255});
  const std::uint8_t *rgbaPixel = rgba.data() + rgba.rowStride() + 2 * 4;
  Image fromRgba = toBgrImage(rgba);
  const std::uint8_t *convertedRgba = fromRgba.ptr<std::uint8_t>(1) + 2 * 3;
  EXPECT_EQ(convertedRgba[0], rgbaPixel[2]);
  EXPECT_EQ(convertedRgba[1], rgbaPixel[1]);
  EXPECT_EQ(convertedRgba[2], rgbaPixel[0]);

  Frame bgra(4, 3, PixelFormat::BGRA8);
  fillPacked(bgra, {10, 20, 30, 255});
  const std::uint8_t *bgraPixel = bgra.data() + bgra.rowStride() + 2 * 4;
  Image fromBgra = toBgrImage(bgra);
  const std::uint8_t *convertedBgra = fromBgra.ptr<std::uint8_t>(1) + 2 * 3;
  EXPECT_EQ(convertedBgra[0], bgraPixel[0]);
  EXPECT_EQ(convertedBgra[1], bgraPixel[1]);
  EXPECT_EQ(convertedBgra[2], bgraPixel[2]);
}

// Neutral chroma (U=V=128) decodes to gray: B=G=R=(298*(Y-16))>>8.
TEST(FrameConversionTest, Yuv420pNeutralChromaDecodesToGray) {
  Frame frame(8, 6, PixelFormat::YUV420P);
  // Luma ramp, chroma neutral.
  for (int y = 0; y < 6; ++y) {
    std::uint8_t *luma = frame.data(0) + y * frame.rowStride(0);
    for (int x = 0; x < 8; ++x) {
      luma[x] = static_cast<std::uint8_t>(16 + 10 * x + 5 * y);
    }
  }
  std::memset(frame.data(1), 128, frame.sizeBytes(1));
  std::memset(frame.data(2), 128, frame.sizeBytes(2));

  Image bgr = toBgrImage(frame);
  ASSERT_EQ(bgr.width(), 8);
  ASSERT_EQ(bgr.height(), 6);
  const int x = 4, y = 3;
  const int expected = (298 * ((16 + 10 * x + 5 * y) - 16)) >> 8;
  const std::uint8_t *pixel = bgr.ptr<std::uint8_t>(y) + x * 3;
  EXPECT_NEAR(pixel[0], expected, 2);
  EXPECT_NEAR(pixel[1], expected, 2);
  EXPECT_NEAR(pixel[2], expected, 2);
}

TEST(FrameConversionTest, Nv12NeutralChromaDecodesToGray) {
  Frame frame(8, 6, PixelFormat::NV12);
  for (int y = 0; y < 6; ++y) {
    std::uint8_t *luma = frame.data(0) + y * frame.rowStride(0);
    for (int x = 0; x < 8; ++x) {
      luma[x] = static_cast<std::uint8_t>(16 + 10 * x + 5 * y);
    }
  }
  std::uint8_t *chroma = frame.data(1);
  for (std::size_t i = 0; i < frame.sizeBytes(1); i += 2) {
    chroma[i] = 128;
    chroma[i + 1] = 128;
  }

  Image bgr = toBgrImage(frame);
  const int x = 4, y = 3;
  const int expected = (298 * ((16 + 10 * x + 5 * y) - 16)) >> 8;
  const std::uint8_t *pixel = bgr.ptr<std::uint8_t>(y) + x * 3;
  EXPECT_NEAR(pixel[0], expected, 2);
  EXPECT_NEAR(pixel[1], expected, 2);
  EXPECT_NEAR(pixel[2], expected, 2);
}

TEST(FrameConversionTest, OddSized420FrameIsRejectedRatherThanMisread) {
  // Frame allows odd dimensions by rounding the chroma planes up, which would
  // leave a chroma row/column without a partner. Refusing beats handing back
  // a plausible-looking picture built from misaligned bytes.
  Frame frame(7, 6, PixelFormat::NV12);
  EXPECT_THROW(toBgrImage(frame), std::invalid_argument);
}

TEST(FrameConversionTest, ToFrameFromBgrImageIsByteIdentical) {
  // A deterministic per-pixel pattern so a stride, size, or channel mix-up
  // cannot pass unnoticed the way a solid color might.
  Image image = Image::uninit(8, 7, 3, PixelType::UInt8);
  for (int y = 0; y < 7; ++y) {
    std::uint8_t *row = image.ptr<std::uint8_t>(y);
    for (int x = 0; x < 8; ++x) {
      row[x * 3] = static_cast<std::uint8_t>(17 + x + 2 * y);
      row[x * 3 + 1] = static_cast<std::uint8_t>(53 + x + 3 * y);
      row[x * 3 + 2] = static_cast<std::uint8_t>(91 + x + 5 * y);
    }
  }

  const videocapture::Frame frame = toFrame(image);
  ASSERT_FALSE(frame.empty());
  EXPECT_EQ(frame.width(), 8);
  EXPECT_EQ(frame.height(), 7);
  EXPECT_EQ(frame.format(), PixelFormat::BGR8);
  EXPECT_EQ(frame.rowStride(), static_cast<std::size_t>(8 * 3));

  for (int y = 0; y < 7; ++y) {
    EXPECT_EQ(0, std::memcmp(frame.data() + y * frame.rowStride(),
                             image.ptr<std::uint8_t>(y),
                             static_cast<std::size_t>(8) * 3))
        << "row " << y << " differs from the source image";
  }
}

TEST(FrameConversionTest, ToFrameRejectsNonPackedBgr8) {
  Image gray = Image::zeros(4, 3, 1, PixelType::UInt8);
  try {
    toFrame(gray);
    FAIL() << "single-channel images must be rejected";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::strstr(error.what(), "toFrame"), nullptr)
        << "rejection should name the toFrame bridging context: "
        << error.what();
  }

  Image floatBgr = Image::zeros(4, 3, 3, PixelType::Float32);
  try {
    toFrame(floatBgr);
    FAIL() << "Float32 images must be rejected";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::strstr(error.what(), "toFrame"), nullptr)
        << "rejection should name the toFrame bridging context: "
        << error.what();
  }
}

TEST(FrameConversionTest, ToFrameFromEmptyImageGivesEmptyFrame) {
  const videocapture::Frame frame = toFrame(Image{});
  EXPECT_TRUE(frame.empty());
  EXPECT_EQ(frame.storageSizeBytes(), 0u);
}

#ifdef VIDEOCAPTURE_WITH_WRITER
TEST(FrameConversionTest, ToFrameRoundTripWritesAndReadsBackAVideo) {
  const std::filesystem::path destination =
      std::filesystem::temp_directory_path() /
      ("neuriplo_infer_writer_roundtrip_" + std::to_string(::getpid()) +
       ".avi");
  const int width = 64;
  const int height = 48;
  const int expectedFrames = 4;

  std::unique_ptr<VideoWriterInterface> writer = createVideoWriter();
  ASSERT_TRUE(writer);
  videocapture::VideoWriterConfig writerConfig;
  writerConfig.width = width;
  writerConfig.height = height;
  writerConfig.frameRate = 30.0;
  writerConfig.codec = videocapture::VideoCodec::Auto;
  if (!writer->initialize(destination.string(), writerConfig)) {
    // Some runners lack any usable encoder; the absence of the codec is an
    // environment property, not a defect in the bridge under test.
    GTEST_SKIP() << "no usable video writer backend on this runner";
  }
  for (int i = 0; i < expectedFrames; ++i) {
    Image image =
        solidImage(width, height, static_cast<std::uint8_t>(i), 90, 160);
    EXPECT_TRUE(writer->writeFrame(neuriplo_infer::toFrame(image)));
  }
  // Checked, not discarded: since videocapture v0.6.0 release() reports
  // whether every accepted frame was encoded and the container finalized, and
  // the re-read below only means something if it was.
  EXPECT_TRUE(writer->release());

  std::unique_ptr<VideoCaptureInterface> reader = createVideoInterface();
  ASSERT_TRUE(reader);
  ASSERT_TRUE(reader->initialize(destination.string()));
  videocapture::Frame frame;
  int readFrames = 0;
  while (reader->readFrame(frame) && !frame.empty()) {
    EXPECT_EQ(frame.width(), width);
    EXPECT_EQ(frame.height(), height);
    ++readFrames;
  }
  reader->release();
  EXPECT_EQ(readFrames, expectedFrames);
  std::filesystem::remove(destination);
}
#endif
