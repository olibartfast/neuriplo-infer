#pragma once

#include "neuriplo/tasks/core/vision/image.hpp"

#include <cstdint>
#include <vector>

namespace neuriplo_infer::imageops {

// App-local image operations replacing the cv::imgproc / cv::core calls the
// renderer and capture paths used to make. Everything works on the
// neuriplo-tasks vision::Image (HxWxC, interleaved, row-major) and BGR channel
// order, which is the order the capture backends and the tensor code assume.
//
// Colour values are BGR to match the rest of the app and cv::Scalar's order.
struct Color {
  std::uint8_t b{0};
  std::uint8_t g{0};
  std::uint8_t r{0};
};

enum class Interpolation { Nearest, Linear };

// Swaps the red and blue channels of a packed 3-channel UInt8 image in place,
// i.e. BGR -> RGB and back. Throws std::invalid_argument for any other shape.
void swapRedBlue(neuriplo_tasks::vision::Image &image);

// Resamples to width x height, preserving channel count and pixel type.
// Bilinear samples use half-pixel centres and clamp at the border; nearest
// takes the floor of the scaled coordinate.
neuriplo_tasks::vision::Image
resize(const neuriplo_tasks::vision::Image &source, int width, int height,
       Interpolation interpolation = Interpolation::Linear);

// Min-max normalises every element into [low, high] and returns a Float32
// image of the same geometry. A constant image maps entirely to `low`.
neuriplo_tasks::vision::Image
normalizeMinMax(const neuriplo_tasks::vision::Image &source, double low = 0.0,
                double high = 1.0);

// Draws a rectangle outline; a thickness <= 0 fills it instead. Out-of-bounds
// geometry is clipped to the image.
void rectangle(neuriplo_tasks::vision::Image &image,
               const neuriplo_tasks::vision::Rect &rect, Color color,
               int thickness = 1);
void fillRectangle(neuriplo_tasks::vision::Image &image,
                   const neuriplo_tasks::vision::Rect &rect, Color color);
void strokeRectangle(neuriplo_tasks::vision::Image &image,
                     const neuriplo_tasks::vision::Rect &rect, Color color,
                     int thickness = 1);

// Anti-aliasing is not reproduced: every primitive rasterises with hard edges.
void line(neuriplo_tasks::vision::Image &image,
          neuriplo_tasks::vision::Point from, neuriplo_tasks::vision::Point to,
          Color color, int thickness = 1);

void fillCircle(neuriplo_tasks::vision::Image &image,
                neuriplo_tasks::vision::Point center, int radius, Color color);
void strokeCircle(neuriplo_tasks::vision::Image &image,
                  neuriplo_tasks::vision::Point center, int radius, Color color,
                  int thickness = 1);

void polylines(
    neuriplo_tasks::vision::Image &image,
    const std::vector<std::vector<neuriplo_tasks::vision::Point>> &contours,
    bool closed, Color color, int thickness = 1);

// Alpha-blends a solid colour over the pixels where `mask` is non-zero:
//   out = out * (1 - alpha) + color * alpha
// `mask` must have the same geometry as `image` and a single channel; both
// UInt8 (any non-zero) and Float32 (greater than zero) masks are accepted.
void maskedAlphaBlend(neuriplo_tasks::vision::Image &image, Color color,
                      const neuriplo_tasks::vision::Image &mask, double alpha);

// TURBO (Google's perceptual rainbow) colour map. `gray` must be single
// channel; values are read as 0..255, so a Float32 image is clamped to that
// range. `destination` is replaced with a 3-channel UInt8 BGR image.
void applyTurboColormap(const neuriplo_tasks::vision::Image &gray,
                        neuriplo_tasks::vision::Image &destination);

} // namespace neuriplo_infer::imageops
