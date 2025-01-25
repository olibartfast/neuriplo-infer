#pragma once
#include <opencv2/opencv.hpp>
#include <string>

bool isDirectory(const std::string& path);
bool isFile(const std::string& path);
void draw_label(cv::Mat& input_image, const std::string& label, float confidence, int left, int top);
std::vector<std::string> readLabelNames(const std::string& fileName);
std::string getFileExtension(const std::string& filename);
std::string getGPUModel();
std::vector<std::string> getGPUModels();
std::string getCPUInfo();
bool hasNvidiaGPU();
std::vector<std::string> split(const std::string& s, char delimiter);
std::vector<std::vector<int64_t>> parseInputSizes(const std::string& input);