#include <fstream>
#include <sstream>
#include <cstring>
#include <cpuid.h>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include "utils.hpp"
namespace fs = std::filesystem;


bool isDirectory(const std::string& path) {
    std::filesystem::path fsPath(path);
    return std::filesystem::is_directory(fsPath);
}

bool isFile(const std::string& path) {
    return std::filesystem::exists(path);
}


void draw_label(cv::Mat& input_image, const std::string& label, float confidence, int left, int top)
{
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = cv::FONT_HERSHEY_DUPLEX; // Change font type to what you think is better for you
    const int THICKNESS = 2;
    cv::Scalar YELLOW = cv::Scalar(0, 255, 255);

    // Display the label and confidence at the top of the bounding box.
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << static_cast<int>(confidence * 100) / 100.0;
    std::string scoreText = out.str();
    
    int baseLine;
    std::string display_text = label + ": " + scoreText;
    cv::Size label_size = cv::getTextSize(display_text, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = cv::max(top, label_size.height);

    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);

    // Draw black rectangle.
    cv::rectangle(input_image, tlc, brc, cv::Scalar(255, 0, 255), cv::FILLED);

    // Put the label and confidence on the black rectangle.
    cv::putText(input_image, display_text, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


std::vector<std::string> readLabelNames(const std::string& fileName)
{
    if(!std::filesystem::exists(fileName)){
        std::cerr << "Wrong path to labels " <<  fileName << std::endl;
        exit(1);
    } 
    std::vector<std::string> classes;
    std::ifstream ifs(fileName.c_str());
    std::string line;
    while (getline(ifs, line))
    classes.push_back(line);
    return classes;   
}

std::string getFileExtension(const std::string& filename) {
    size_t dotPos = filename.find_last_of(".");
    if (dotPos != std::string::npos) {
        return filename.substr(dotPos + 1);
    }
    return ""; // Return empty string if no extension found
}
std::vector<std::string> getGPUModels() {
    constexpr auto GPU_DIRECTORY = "/proc/driver/nvidia/gpus";
    std::vector<std::string> gpuModels;

    if (!fs::exists(GPU_DIRECTORY)) {
        throw std::runtime_error("NVIDIA GPU directory not found");
    }

    for (const auto& directory : fs::directory_iterator(GPU_DIRECTORY)) {
        std::ifstream gpuInfoStream(directory.path() / "information");
        if (!gpuInfoStream) {
            continue;  // Skip if unable to open the file
        }

        std::string gpuInfoLine;
        while (std::getline(gpuInfoStream, gpuInfoLine)) {
            if (gpuInfoLine.find("Model:") != std::string::npos) {
                size_t colonPos = gpuInfoLine.find(':');
                if (colonPos != std::string::npos) {
                    std::string gpuModel = gpuInfoLine.substr(colonPos + 1);
                    gpuModel.erase(0, gpuModel.find_first_not_of(" \t"));
                    gpuModel.erase(gpuModel.find_last_not_of(" \t") + 1);
                    gpuModels.push_back(std::move(gpuModel));
                    break;
                }
            }
        }
        gpuInfoStream.close();
    }

    if (gpuModels.empty()) {
        throw std::runtime_error("No GPU models found");
    }

    return gpuModels;
}

// Usage example
std::string getGPUModel() {
    try {
        auto models = getGPUModels();
        return models.front();  // Return the first GPU model
    } catch (const std::exception& e) {
        // Log the error or handle it as appropriate for your application
        return "Unknown GPU";
    }
}

std::string getCPUInfo() {
    std::string cpuInfo;

    unsigned int cpuInfoRegs[4];
    __cpuid(0x80000000, cpuInfoRegs[0], cpuInfoRegs[1], cpuInfoRegs[2], cpuInfoRegs[3]);
    unsigned int extMaxId = cpuInfoRegs[0];

    char brand[48];
    if (extMaxId >= 0x80000004) {
        __cpuid(0x80000002, cpuInfoRegs[0], cpuInfoRegs[1], cpuInfoRegs[2], cpuInfoRegs[3]);
        memcpy(brand, cpuInfoRegs, sizeof(cpuInfoRegs));
        __cpuid(0x80000003, cpuInfoRegs[0], cpuInfoRegs[1], cpuInfoRegs[2], cpuInfoRegs[3]);
        memcpy(brand + 16, cpuInfoRegs, sizeof(cpuInfoRegs));
        __cpuid(0x80000004, cpuInfoRegs[0], cpuInfoRegs[1], cpuInfoRegs[2], cpuInfoRegs[3]);
        memcpy(brand + 32, cpuInfoRegs, sizeof(cpuInfoRegs));
        cpuInfo = brand;
    }

    return cpuInfo;
}


bool hasNvidiaGPU() {
    const std::vector<std::string> nvidiaIndicators = {
        "/proc/driver/nvidia",
        "/dev/nvidia0",
        "/sys/class/nvidia-gpu"
    };

    for (const auto& path : nvidiaIndicators) {
        if (fs::exists(path)) {
            return true;
        }
    }

    return false;
}