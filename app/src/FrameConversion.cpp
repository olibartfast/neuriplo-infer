#include "FrameConversion.hpp"

#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace neuriplo_infer {

namespace {

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

// A Mat header over one plane. No pixels are copied, and the frame's own row
// stride is passed through, so a plane with padded rows maps correctly instead
// of being read as if it were tightly packed.
cv::Mat planeHeader(Frame &frame, int type, std::size_t plane = 0) {
  return cv::Mat(frame.planeHeight(plane), frame.planeWidth(plane), type,
                 frame.data(plane), frame.rowStride(plane));
}

// OpenCV decodes NV12 and YUV420P from a single-channel buffer holding the
// luma rows followed by the chroma rows, all at the luma row pitch. That is
// exactly Frame's canonical layout for even dimensions -- but only for even
// dimensions, and only while the planes stay tightly packed and adjacent, so
// verify rather than assume. Getting this wrong would not fail loudly; it
// would silently produce a picture with the colors torn out of place.
cv::Mat planarLumaChromaHeader(Frame &frame,
                               const std::vector<std::size_t> &chromaStrides) {
  const int width = frame.width();
  const int height = frame.height();
  if (width % 2 != 0 || height % 2 != 0) {
    throw std::invalid_argument(
        "videocapture::Frame in " + formatName(frame.format()) + " is " +
        std::to_string(width) + "x" + std::to_string(height) +
        "; planar 4:2:0 conversion requires even dimensions");
  }

  if (frame.rowStride(0) != static_cast<std::size_t>(width)) {
    throw std::invalid_argument(
        "videocapture::Frame in " + formatName(frame.format()) +
        " has a padded luma plane; planar 4:2:0 conversion requires a tightly "
        "packed layout");
  }

  const std::uint8_t *expected = frame.data(0) + frame.sizeBytes(0);
  for (std::size_t plane = 1; plane <= chromaStrides.size(); ++plane) {
    if (frame.rowStride(plane) != chromaStrides[plane - 1] ||
        frame.data(plane) != expected) {
      throw std::invalid_argument(
          "videocapture::Frame in " + formatName(frame.format()) +
          " does not use the tightly packed plane layout that planar 4:2:0 "
          "conversion requires");
    }
    expected += frame.sizeBytes(plane);
  }

  // Luma rows plus the chroma rows that follow, both at the luma pitch.
  return cv::Mat(height + height / 2, width, CV_8UC1, frame.data(0),
                 static_cast<std::size_t>(width));
}

} // namespace

namespace {

// Describes a Mat's pixel type the way toBgrMat's rejection messages do, so
// the failure names the actual state rather than just "unsupported".
std::string cvMatTypeDescription(int type) {
  const int depth = CV_MAT_DEPTH(type);
  switch (depth) {
  case CV_8U: return "CV_8U";
  case CV_8S: return "CV_8S";
  case CV_16U: return "CV_16U";
  case CV_16S: return "CV_16S";
  case CV_32S: return "CV_32S";
  case CV_32F: return "CV_32F";
  case CV_64F: return "CV_64F";
  default: return "unknown";
  }
}

} // namespace

cv::Mat toBgrMat(Frame &frame) {
  if (frame.empty()) {
    return {};
  }

  cv::Mat converted;
  switch (frame.format()) {
  case PixelFormat::BGR8:
    // The only zero-copy case, and the one every capture backend produces
    // today: already the layout the rest of the app works in.
    return planeHeader(frame, CV_8UC3);
  case PixelFormat::RGB8:
    cv::cvtColor(planeHeader(frame, CV_8UC3), converted, cv::COLOR_RGB2BGR);
    return converted;
  case PixelFormat::Gray8:
    cv::cvtColor(planeHeader(frame, CV_8UC1), converted, cv::COLOR_GRAY2BGR);
    return converted;
  case PixelFormat::RGBA8:
    cv::cvtColor(planeHeader(frame, CV_8UC4), converted, cv::COLOR_RGBA2BGR);
    return converted;
  case PixelFormat::BGRA8:
    cv::cvtColor(planeHeader(frame, CV_8UC4), converted, cv::COLOR_BGRA2BGR);
    return converted;
  case PixelFormat::NV12: {
    const std::size_t width = static_cast<std::size_t>(frame.width());
    cv::cvtColor(planarLumaChromaHeader(frame, {width}), converted,
                 cv::COLOR_YUV2BGR_NV12);
    return converted;
  }
  case PixelFormat::YUV420P: {
    const std::size_t chroma = static_cast<std::size_t>(frame.width()) / 2;
    cv::cvtColor(planarLumaChromaHeader(frame, {chroma, chroma}), converted,
                 cv::COLOR_YUV2BGR_I420);
    return converted;
  }
  }

  throw std::invalid_argument("videocapture::Frame carries an unrecognized "
                              "pixel format and cannot be converted to BGR");
}

videocapture::Frame toFrame(const cv::Mat &mat) {
  if (mat.empty()) {
    return {};
  }

  if (mat.type() != CV_8UC3) {
    throw std::invalid_argument(
        "cv::Mat of type " + cvMatTypeDescription(mat.type()) +
        " cannot be bridged to a BGR8 videocapture::Frame; packed CV_8UC3 "
        "is required");
  }

  if (!mat.isContinuous()) {
    throw std::invalid_argument(
        "cv::Mat bridged to a videocapture::Frame must be continuous; the "
        "renderer produces tightly packed rows, so clone the Mat before "
        "calling toFrame");
  }

  const int width = mat.cols;
  const int height = mat.rows;
  videocapture::Frame frame(width, height, videocapture::PixelFormat::BGR8);
  // Frame's canonical BGR8 row stride is width*3 and the Mat is continuous
  // packed BGR8, so the rows copy byte-identically.
  for (int y = 0; y < height; ++y) {
    const std::uint8_t *source = mat.data + y * width * 3;
    std::memcpy(frame.data() + y * frame.rowStride(), source,
                static_cast<std::size_t>(width) * 3);
  }
  return frame;
}

} // namespace neuriplo_infer
