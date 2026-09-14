#pragma once

#include "Frame.hpp"

#include <opencv2/core.hpp>

namespace neuriplo_infer {

// videocapture v0.4.0 moved VideoCaptureInterface::readFrame() off cv::Mat and
// onto videocapture::Frame, a dependency-free carrier with an explicit pixel
// format, per-plane row strides, and a timestamp. The app pipeline is OpenCV
// end to end -- preprocessing, rendering, and display all take cv::Mat -- so
// frames are bridged here, at the single point video enters the app, rather
// than teaching every stage a second image type.
//
// Returns a BGR8 cv::Mat, the layout every downstream stage already assumes.
//
// On the common path (a packed BGR8 frame, which is what all three capture
// backends currently produce) the result is a header *aliasing* the frame's
// own storage: no pixels are copied, and writes through the Mat -- the FPS
// overlay, the rendered detections -- land in the frame's buffer. Two
// consequences follow from that alias:
//
//   * The Mat is valid only until the next readFrame() on the same frame.
//     Frame::resize() may reallocate, which leaves any earlier Mat dangling,
//     so convert after each read rather than hoisting the Mat out of the loop.
//   * Anything that must outlive the next read needs its own copy (.clone()).
//
// Any other pixel format is converted into a freshly allocated buffer, so the
// Mat owns its pixels and writes do not reach the frame. Callers that only
// read, render, and display -- which is all of them today -- need not care
// which of the two they were handed.
//
// An empty frame converts to an empty Mat. A frame whose planar layout is not
// the canonical tightly packed one throws std::invalid_argument rather than
// reinterpreting the bytes as something they are not.
cv::Mat toBgrMat(videocapture::Frame &frame);

// Inverse of toBgrMat, the writer bridge: hands a rendered Mat to videocapture
// APIs that consume Frame instead of cv::Mat, e.g. a video writer sink.
// Always returns a Frame that owns its own storage (a copy of the Mat's
// bytes) -- unlike the packed-BGR8 path back from a Frame, there is no alias.
//
// The Mat must already be continuous packed BGR8 (CV_8UC3), which is what the
// renderer produces. An empty Mat yields an empty Frame; anything else --
// other depths, other channel counts, non-contiguous storage -- throws
// std::invalid_argument naming the actual type and state.
videocapture::Frame toFrame(const cv::Mat &mat);

} // namespace neuriplo_infer
