#pragma once

#include "AppImageOps.hpp"
#include "neuriplo/tasks/core/vision/image.hpp"

#include <string>

bool isDirectory(const std::string &path);
bool isFile(const std::string &path);
// Filled band marking a detection, sized to the label text it now carries.
// Black bitmap text is drawn on the band; overlong labels are clipped by
// the image edge rather than wrapped.
void drawLabelBand(neuriplo_tasks::vision::Image &image, int left, int top,
                   const std::string &text,
                   neuriplo_infer::imageops::Color color);

// Integer glyph scale for overlays, proportional to the frame's shorter
// side (1 at <=360 px, 3 at 1080p, capped by legibility rather than size).
int labelTextScale(const neuriplo_tasks::vision::Image &image);
std::vector<std::string> readLabelNames(const std::string &fileName);
std::string getFileExtension(const std::string &filename);
// A source argument naming a still image, judged by its extension
// case-insensitively. Routing and argument validation share it so they can
// never disagree about what is an image.
bool isStillImageSource(const std::string &path);
std::string getGPUModel();
std::vector<std::string> getGPUModels();
std::string getCPUInfo();
bool hasNvidiaGPU();
std::vector<std::string> split(const std::string &s, char delimiter);
std::vector<std::vector<int64_t>> parseInputSizes(const std::string &input);
std::string normalizeModelType(const std::string &model_type);
