#include "FrameConversion.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace neuriplo_infer {

namespace {

using neuriplo_tasks::vision::Image;
using neuriplo_tasks::vision::PixelType;
using videocapture::Frame;
using videocapture::PixelFormat;

std::string formatName(PixelFormat format) {
  switch (format) {
  case PixelFormat::Gray8:
    return "Gray8";
  case PixelFormat::RGB8:
    return "RGB8";
  case PixelFormat::BGR8:
    return "BGR8";
  case PixelFormat::RGBA8:
    return "RGBA8";
  case PixelFormat::BGRA8:
    return "BGRA8";
  case PixelFormat::NV12:
    return "NV12";
  case PixelFormat::YUV420P:
    return "YUV420P";
  }
  return "unknown";
}

std::string pixelTypeName(PixelType pixelType) {
  switch (pixelType) {
  case PixelType::UInt8:
    return "UInt8";
  case PixelType::Float32:
    return "Float32";
  case PixelType::Int32:
    return "Int32";
  }
  return "unknown";
}

const std::uint8_t *planeRow(const Frame &frame, std::size_t plane, int row) {
  return frame.data(plane) +
         static_cast<std::size_t>(row) * frame.rowStride(plane);
}

std::uint8_t clampByte(int value) {
  return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

// BT.601 limited-range YUV to BGR, matching the conversion the app previously
// delegated to cv::cvtColor's YUV420/NV12 paths.
void writeYuvPixel(std::uint8_t *destination, int y, int u, int v) {
  const int c = y - 16;
  const int d = u - 128;
  const int e = v - 128;
  destination[0] = clampByte((298 * c + 516 * d + 128) >> 8);
  destination[1] = clampByte((298 * c - 100 * d - 208 * e + 128) >> 8);
  destination[2] = clampByte((298 * c + 409 * e + 128) >> 8);
}

// NV12 keeps U and V interleaved in one chroma plane, YUV420P in two. Both are
// 4:2:0, so every 2x2 luma block shares one chroma pair, and odd dimensions
// would leave a chroma row/column without a partner.
void convertPlanar420(const Frame &frame, Image &destination,
                      bool interleaved) {
  const int width = frame.width();
  const int height = frame.height();
  if (width % 2 != 0 || height % 2 != 0) {
    throw std::invalid_argument(
        "videocapture::Frame in " + formatName(frame.format()) + " is " +
        std::to_string(width) + "x" + std::to_string(height) +
        "; planar 4:2:0 conversion requires even dimensions");
  }

  for (int y = 0; y < height; ++y) {
    const std::uint8_t *luma = planeRow(frame, 0, y);
    const std::uint8_t *uPlane = planeRow(frame, 1, y / 2);
    const std::uint8_t *vPlane =
        interleaved ? uPlane : planeRow(frame, 2, y / 2);
    std::uint8_t *row = destination.ptr<std::uint8_t>(y);

    for (int x = 0; x < width; ++x) {
      const int chroma = x / 2;
      const int u = interleaved ? uPlane[2 * chroma] : uPlane[chroma];
      const int v = interleaved ? vPlane[2 * chroma + 1] : vPlane[chroma];
      writeYuvPixel(&row[3 * static_cast<std::size_t>(x)], luma[x], u, v);
    }
  }
}

} // namespace

neuriplo_tasks::vision::Image toBgrImage(videocapture::Frame &frame) {
  if (frame.empty()) {
    return {};
  }

  const int width = frame.width();
  const int height = frame.height();
  const std::size_t rowBytes = static_cast<std::size_t>(width) * 3;

  switch (frame.format()) {
  case PixelFormat::BGR8: {
    Image image = Image::uninit(width, height, 3, PixelType::UInt8);
    for (int y = 0; y < height; ++y) {
      std::memcpy(image.ptr<std::uint8_t>(y), planeRow(frame, 0, y), rowBytes);
    }
    return image;
  }
  case PixelFormat::RGB8: {
    Image image = Image::uninit(width, height, 3, PixelType::UInt8);
    for (int y = 0; y < height; ++y) {
      const std::uint8_t *source = planeRow(frame, 0, y);
      std::uint8_t *row = image.ptr<std::uint8_t>(y);
      for (int x = 0; x < width; ++x) {
        const std::size_t pixel = 3 * static_cast<std::size_t>(x);
        row[pixel] = source[pixel + 2];
        row[pixel + 1] = source[pixel + 1];
        row[pixel + 2] = source[pixel];
      }
    }
    return image;
  }
  case PixelFormat::Gray8: {
    Image image = Image::uninit(width, height, 3, PixelType::UInt8);
    for (int y = 0; y < height; ++y) {
      const std::uint8_t *source = planeRow(frame, 0, y);
      std::uint8_t *row = image.ptr<std::uint8_t>(y);
      for (int x = 0; x < width; ++x) {
        const std::size_t pixel = 3 * static_cast<std::size_t>(x);
        row[pixel] = source[x];
        row[pixel + 1] = source[x];
        row[pixel + 2] = source[x];
      }
    }
    return image;
  }
  case PixelFormat::RGBA8:
  case PixelFormat::BGRA8: {
    // Red and blue are the only channels that move: dropping the trailing
    // alpha leaves RGBA -> BGR (swap) and BGRA -> BGR (already in order).
    const bool swapRedBlue = frame.format() == PixelFormat::RGBA8;
    Image image = Image::uninit(width, height, 3, PixelType::UInt8);
    for (int y = 0; y < height; ++y) {
      const std::uint8_t *source = planeRow(frame, 0, y);
      std::uint8_t *row = image.ptr<std::uint8_t>(y);
      for (int x = 0; x < width; ++x) {
        const std::size_t pixel = 3 * static_cast<std::size_t>(x);
        const std::size_t sourcePixel = 4 * static_cast<std::size_t>(x);
        row[pixel] = source[sourcePixel + (swapRedBlue ? 2 : 0)];
        row[pixel + 1] = source[sourcePixel + 1];
        row[pixel + 2] = source[sourcePixel + (swapRedBlue ? 0 : 2)];
      }
    }
    return image;
  }
  case PixelFormat::NV12: {
    Image image = Image::uninit(width, height, 3, PixelType::UInt8);
    convertPlanar420(frame, image, true);
    return image;
  }
  case PixelFormat::YUV420P: {
    Image image = Image::uninit(width, height, 3, PixelType::UInt8);
    convertPlanar420(frame, image, false);
    return image;
  }
  }

  throw std::invalid_argument("videocapture::Frame carries an unrecognized "
                              "pixel format and cannot be converted to BGR");
}

videocapture::Frame toFrame(const neuriplo_tasks::vision::Image &image) {
  if (image.empty()) {
    return {};
  }

  if (image.channels() != 3 || image.pixelType() != PixelType::UInt8) {
    throw std::invalid_argument(
        "toFrame requires a packed BGR UInt8 image with 3 channels; got " +
        pixelTypeName(image.pixelType()) + " with " +
        std::to_string(image.channels()) + " channels");
  }

  const int width = image.width();
  const int height = image.height();
  const std::size_t rowBytes = static_cast<std::size_t>(width) * 3;

  videocapture::Frame frame(width, height, videocapture::PixelFormat::BGR8);
  for (int y = 0; y < height; ++y) {
    std::memcpy(frame.data() + static_cast<std::size_t>(y) * frame.rowStride(),
                image.ptr<std::uint8_t>(y), rowBytes);
  }
  return frame;
}

} // namespace neuriplo_infer
