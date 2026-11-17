#pragma once

#include "AppImageOps.hpp"
#include "neuriplo/tasks/core/vision/image.hpp"

#include <string>

bool isDirectory(const std::string &path);
bool isFile(const std::string &path);
// Filled confidence band marking a detection. Text rendering has no
// OpenCV-free equivalent in the app, so the label string and confidence value
// are intentionally not drawn; the band's position and colour still identify
// which box produced it.
void drawLabelBand(neuriplo_tasks::vision::Image &image, int left, int top,
                   neuriplo_infer::imageops::Color color);
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
