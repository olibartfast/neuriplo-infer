#pragma once

#include "Frame.hpp"

#include "neuriplo/tasks/core/vision/image.hpp"

namespace neuriplo_infer {

// videocapture v0.4.0 moved VideoCaptureInterface::readFrame() off cv::Mat and
// onto videocapture::Frame, a dependency-free carrier with an explicit pixel
// format, per-plane row strides, and a timestamp. The app pipeline now works
// on neuriplo-tasks vision::Image (HxWxC, tightly packed, BGR for colour), so
// frames are bridged here, at the single point video enters the app.
//
// Returns a BGR UInt8, 3-channel image. Unlike the old cv::Mat bridge this
// always owns its pixels: Image has no aliasing constructor, so the packed
// BGR8 fast path is a row-wise copy. Every other format is converted into a
// freshly allocated buffer as well.
//
// An empty frame converts to an empty image. Planar 4:2:0 formats require
// even dimensions; odd ones throw std::invalid_argument rather than
// reinterpreting the bytes as something they are not.
neuriplo_tasks::vision::Image toBgrImage(videocapture::Frame &frame);

// Inverse of toBgrImage, the writer bridge: hands a rendered image to
// videocapture APIs that consume Frame instead of cv::Mat.
//
// The image must be packed BGR UInt8 (3 channels), which is what the renderer
// produces. An empty image yields an empty Frame; anything else throws
// std::invalid_argument naming the actual type and state.
videocapture::Frame toFrame(const neuriplo_tasks::vision::Image &image);

} // namespace neuriplo_infer
