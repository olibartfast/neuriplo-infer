#include "AppImageOps.hpp"
#include "BitmapFont.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace neuriplo_infer::imageops {

namespace {

using neuriplo_tasks::vision::Image;
using neuriplo_tasks::vision::PixelType;

std::uint8_t clampByte(double value) {
  return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

double readPixel(const Image &image, int x, int y, int channel) {
  const std::size_t index =
      (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width()) +
       static_cast<std::size_t>(x)) *
          static_cast<std::size_t>(image.channels()) +
      static_cast<std::size_t>(channel);
  switch (image.pixelType()) {
  case PixelType::UInt8:
    return image.data<std::uint8_t>()[index];
  case PixelType::Float32:
    return image.data<float>()[index];
  case PixelType::Int32:
    return image.data<std::int32_t>()[index];
  }
  return 0.0;
}

void writePixel(Image &image, int x, int y, int channel, double value) {
  const std::size_t index =
      (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width()) +
       static_cast<std::size_t>(x)) *
          static_cast<std::size_t>(image.channels()) +
      static_cast<std::size_t>(channel);
  switch (image.pixelType()) {
  case PixelType::UInt8:
    image.data<std::uint8_t>()[index] = clampByte(value);
    break;
  case PixelType::Float32:
    image.data<float>()[index] = static_cast<float>(value);
    break;
  case PixelType::Int32:
    image.data<std::int32_t>()[index] =
        static_cast<std::int32_t>(std::lround(value));
    break;
  }
}

bool inside(const Image &image, int x, int y) {
  return x >= 0 && y >= 0 && x < image.width() && y < image.height();
}

void putPixel(Image &image, int x, int y, Color color) {
  if (!inside(image, x, y) || image.channels() < 3 ||
      image.pixelType() != PixelType::UInt8) {
    return;
  }
  std::uint8_t *pixel =
      image.ptr<std::uint8_t>(y) + static_cast<std::size_t>(x) * 3;
  pixel[0] = color.b;
  pixel[1] = color.g;
  pixel[2] = color.r;
}

void drawThickPoint(Image &image, int center_x, int center_y, int thickness,
                    Color color) {
  const int radius = std::max(0, thickness / 2);
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      putPixel(image, center_x + dx, center_y + dy, color);
    }
  }
}

void blendPixel(Image &image, int x, int y, Color color, double alpha) {
  if (!inside(image, x, y) || image.channels() < 3 ||
      image.pixelType() != PixelType::UInt8) {
    return;
  }
  std::uint8_t *pixel =
      image.ptr<std::uint8_t>(y) + static_cast<std::size_t>(x) * 3;
  const double keep = 1.0 - alpha;
  pixel[0] = clampByte(static_cast<double>(pixel[0]) * keep +
                       static_cast<double>(color.b) * alpha);
  pixel[1] = clampByte(static_cast<double>(pixel[1]) * keep +
                       static_cast<double>(color.g) * alpha);
  pixel[2] = clampByte(static_cast<double>(pixel[2]) * keep +
                       static_cast<double>(color.r) * alpha);
}

const std::array<Color, 256> &turboLut() {
  static const std::array<Color, 256> lut = [] {
    std::array<Color, 256> table{};
    for (std::size_t i = 0; i < table.size(); ++i) {
      const double t = static_cast<double>(i) / 255.0;
      const double t2 = t * t;
      const double t3 = t2 * t;
      const double t4 = t3 * t;
      const double t5 = t4 * t;
      const double red = 0.13572138 + 4.61539260 * t - 42.66032258 * t2 +
                         132.13108234 * t3 - 152.94239396 * t4 +
                         59.28637943 * t5;
      const double green = 0.09140261 + 2.19418839 * t + 4.84296658 * t2 -
                           14.18503333 * t3 + 4.27729857 * t4 + 2.82956604 * t5;
      const double blue = 0.10667330 + 12.64194608 * t - 60.58204836 * t2 +
                          110.36276771 * t3 - 89.90310912 * t4 +
                          27.34824973 * t5;
      table[i] = Color{clampByte(blue * 255.0), clampByte(green * 255.0),
                       clampByte(red * 255.0)};
    }
    return table;
  }();
  return lut;
}

} // namespace

void swapRedBlue(Image &image) {
  if (image.empty()) {
    return;
  }
  if (image.channels() != 3 || image.pixelType() != PixelType::UInt8) {
    throw std::invalid_argument(
        "swapRedBlue requires a packed 3-channel UInt8 image");
  }
  for (int y = 0; y < image.height(); ++y) {
    std::uint8_t *row = image.ptr<std::uint8_t>(y);
    for (int x = 0; x < image.width(); ++x) {
      std::swap(row[static_cast<std::size_t>(x) * 3],
                row[static_cast<std::size_t>(x) * 3 + 2]);
    }
  }
}

Image resize(const Image &source, int width, int height,
             Interpolation interpolation) {
  if (source.empty() || width <= 0 || height <= 0) {
    return {};
  }

  Image destination =
      Image::uninit(width, height, source.channels(), source.pixelType());
  const double scale_x =
      static_cast<double>(source.width()) / static_cast<double>(width);
  const double scale_y =
      static_cast<double>(source.height()) / static_cast<double>(height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int channel = 0; channel < source.channels(); ++channel) {
        if (interpolation == Interpolation::Nearest) {
          const int source_x =
              std::min(source.width() - 1,
                       static_cast<int>(static_cast<double>(x) * scale_x));
          const int source_y =
              std::min(source.height() - 1,
                       static_cast<int>(static_cast<double>(y) * scale_y));
          writePixel(destination, x, y, channel,
                     readPixel(source, source_x, source_y, channel));
          continue;
        }

        const double source_x = (static_cast<double>(x) + 0.5) * scale_x - 0.5;
        const double source_y = (static_cast<double>(y) + 0.5) * scale_y - 0.5;
        const int x0 = static_cast<int>(std::floor(source_x));
        const int y0 = static_cast<int>(std::floor(source_y));
        const double fx = source_x - static_cast<double>(x0);
        const double fy = source_y - static_cast<double>(y0);
        const int x1 = std::clamp(x0 + 1, 0, source.width() - 1);
        const int y1 = std::clamp(y0 + 1, 0, source.height() - 1);
        const int cx0 = std::clamp(x0, 0, source.width() - 1);
        const int cy0 = std::clamp(y0, 0, source.height() - 1);

        const double top = readPixel(source, cx0, cy0, channel) * (1.0 - fx) +
                           readPixel(source, x1, cy0, channel) * fx;
        const double bottom = readPixel(source, cx0, y1, channel) * (1.0 - fx) +
                              readPixel(source, x1, y1, channel) * fx;
        writePixel(destination, x, y, channel, top * (1.0 - fy) + bottom * fy);
      }
    }
  }
  return destination;
}

Image normalizeMinMax(const Image &source, double low, double high) {
  if (source.empty()) {
    return {};
  }

  const std::size_t count =
      source.totalPixels() * static_cast<std::size_t>(source.channels());
  double minimum = readPixel(source, 0, 0, 0);
  double maximum = minimum;
  for (std::size_t i = 0; i < count; ++i) {
    const int channel =
        static_cast<int>(i % static_cast<std::size_t>(source.channels()));
    const int x =
        static_cast<int>((i / static_cast<std::size_t>(source.channels())) %
                         static_cast<std::size_t>(source.width()));
    const int y =
        static_cast<int>((i / static_cast<std::size_t>(source.channels())) /
                         static_cast<std::size_t>(source.width()));
    const double value = readPixel(source, x, y, channel);
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }

  Image destination = Image::uninit(source.width(), source.height(),
                                    source.channels(), PixelType::Float32);
  const double range = maximum - minimum;
  for (int y = 0; y < source.height(); ++y) {
    for (int x = 0; x < source.width(); ++x) {
      for (int channel = 0; channel < source.channels(); ++channel) {
        const double value = readPixel(source, x, y, channel);
        const double scaled =
            range == 0.0 ? low : low + (value - minimum) / range * (high - low);
        writePixel(destination, x, y, channel, scaled);
      }
    }
  }
  return destination;
}

void rectangle(Image &image, const neuriplo_tasks::vision::Rect &rect,
               Color color, int thickness) {
  if (thickness <= 0) {
    fillRectangle(image, rect, color);
    return;
  }
  strokeRectangle(image, rect, color, thickness);
}

void fillRectangle(Image &image, const neuriplo_tasks::vision::Rect &rect,
                   Color color) {
  const int x1 = std::clamp(rect.x, 0, image.width());
  const int y1 = std::clamp(rect.y, 0, image.height());
  const int x2 = std::clamp(rect.x + rect.width, 0, image.width());
  const int y2 = std::clamp(rect.y + rect.height, 0, image.height());
  for (int y = y1; y < y2; ++y) {
    for (int x = x1; x < x2; ++x) {
      putPixel(image, x, y, color);
    }
  }
}

void strokeRectangle(Image &image, const neuriplo_tasks::vision::Rect &rect,
                     Color color, int thickness) {
  const int right = rect.x + rect.width - 1;
  const int bottom = rect.y + rect.height - 1;
  line(image, {rect.x, rect.y}, {right, rect.y}, color, thickness);
  line(image, {right, rect.y}, {right, bottom}, color, thickness);
  line(image, {right, bottom}, {rect.x, bottom}, color, thickness);
  line(image, {rect.x, bottom}, {rect.x, rect.y}, color, thickness);
}

void line(Image &image, neuriplo_tasks::vision::Point from,
          neuriplo_tasks::vision::Point to, Color color, int thickness) {
  int x0 = from.x;
  int y0 = from.y;
  const int x1 = to.x;
  const int y1 = to.y;
  const int dx = std::abs(x1 - x0);
  const int dy = -std::abs(y1 - y0);
  const int step_x = x0 < x1 ? 1 : -1;
  const int step_y = y0 < y1 ? 1 : -1;
  int error = dx + dy;
  while (true) {
    drawThickPoint(image, x0, y0, thickness, color);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    const int doubled = 2 * error;
    if (doubled >= dy) {
      error += dy;
      x0 += step_x;
    }
    if (doubled <= dx) {
      error += dx;
      y0 += step_y;
    }
  }
}

void fillCircle(Image &image, neuriplo_tasks::vision::Point center, int radius,
                Color color) {
  for (int dy = -radius; dy <= radius; ++dy) {
    const int span = static_cast<int>(
        std::sqrt(static_cast<double>(radius * radius - dy * dy)));
    for (int dx = -span; dx <= span; ++dx) {
      putPixel(image, center.x + dx, center.y + dy, color);
    }
  }
}

void strokeCircle(Image &image, neuriplo_tasks::vision::Point center,
                  int radius, Color color, int thickness) {
  if (radius <= 0) {
    drawThickPoint(image, center.x, center.y, thickness, color);
    return;
  }
  const int steps = std::max(8, 8 * radius);
  for (int step = 0; step < steps; ++step) {
    const double angle = 2.0 * 3.14159265358979323846 *
                         static_cast<double>(step) / static_cast<double>(steps);
    drawThickPoint(
        image,
        center.x + static_cast<int>(std::lround(static_cast<double>(radius) *
                                                std::cos(angle))),
        center.y + static_cast<int>(std::lround(static_cast<double>(radius) *
                                                std::sin(angle))),
        thickness, color);
  }
}

void polylines(
    Image &image,
    const std::vector<std::vector<neuriplo_tasks::vision::Point>> &contours,
    bool closed, Color color, int thickness) {
  for (const auto &contour : contours) {
    if (contour.empty()) {
      continue;
    }
    for (std::size_t i = 1; i < contour.size(); ++i) {
      line(image, contour[i - 1], contour[i], color, thickness);
    }
    if (closed && contour.size() >= 2) {
      line(image, contour.back(), contour.front(), color, thickness);
    }
  }
}

void maskedAlphaBlend(Image &image, Color color, const Image &mask,
                      double alpha) {
  if (mask.width() != image.width() || mask.height() != image.height() ||
      mask.channels() != 1) {
    throw std::invalid_argument(
        "maskedAlphaBlend requires a single-channel mask matching the "
        "image geometry");
  }
  const double clamped = std::clamp(alpha, 0.0, 1.0);
  for (int y = 0; y < image.height(); ++y) {
    for (int x = 0; x < image.width(); ++x) {
      if (readPixel(mask, x, y, 0) > 0.0) {
        blendPixel(image, x, y, color, clamped);
      }
    }
  }
}

void applyTurboColormap(const Image &gray, Image &destination) {
  if (gray.empty()) {
    destination = {};
    return;
  }
  if (gray.channels() != 1) {
    throw std::invalid_argument(
        "applyTurboColormap requires a single-channel image");
  }

  destination = Image::uninit(gray.width(), gray.height(), 3, PixelType::UInt8);
  const auto &lut = turboLut();
  for (int y = 0; y < gray.height(); ++y) {
    std::uint8_t *row = destination.ptr<std::uint8_t>(y);
    for (int x = 0; x < gray.width(); ++x) {
      const int index =
          static_cast<int>(std::clamp(readPixel(gray, x, y, 0), 0.0, 255.0));
      const Color &color = lut[static_cast<std::size_t>(index)];
      row[static_cast<std::size_t>(x) * 3] = color.b;
      row[static_cast<std::size_t>(x) * 3 + 1] = color.g;
      row[static_cast<std::size_t>(x) * 3 + 2] = color.r;
    }
  }
}

neuriplo_tasks::vision::Point textSize(const std::string &text, int scale) {
  if (text.empty() || scale <= 0) {
    return neuriplo_tasks::vision::Point{0, 0};
  }
  const int glyph_scale = std::max(1, scale);
  return neuriplo_tasks::vision::Point{
      static_cast<int>(bitmap_font::kAdvance * glyph_scale * text.size()) -
          glyph_scale,
      bitmap_font::kHeight * glyph_scale};
}

void drawText(Image &image, const std::string &text, int x, int y, Color color,
              int scale) {
  // Text is a render-stage overlay: silently draw nothing on images that
  // cannot carry it rather than failing the run.
  if (text.empty() || image.channels() < 3 ||
      image.pixelType() != PixelType::UInt8) {
    return;
  }
  const int glyph_scale = std::max(1, scale);
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char *const *rows =
        bitmap_font::glyphFor(static_cast<unsigned char>(text[i]));
    if (rows == nullptr) {
      continue;
    }
    const int origin_x =
        x + static_cast<int>(i) * bitmap_font::kAdvance * glyph_scale;
    for (int row = 0; row < bitmap_font::kHeight; ++row) {
      for (int col = 0; col < bitmap_font::kWidth; ++col) {
        if (rows[row][col] != 'X') {
          continue;
        }
        for (int dy = 0; dy < glyph_scale; ++dy) {
          for (int dx = 0; dx < glyph_scale; ++dx) {
            putPixel(image, origin_x + col * glyph_scale + dx,
                     y + row * glyph_scale + dy, color);
          }
        }
      }
    }
  }
}

} // namespace neuriplo_infer::imageops
