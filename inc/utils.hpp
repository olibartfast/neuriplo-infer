#pragma once
#include "common.hpp"


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
    int baseLine;
    std::string display_text = label + ": " + std::to_string(confidence);
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

std::vector<float> mat2vector(const cv::Mat& input_blob)
{

    const auto channels = input_blob.channels();
    const auto network_width = input_blob.cols;
    const auto network_height = input_blob.rows;
    size_t img_byte_size = input_blob.total() * input_blob.elemSize();  // Allocate a buffer to hold all image elements.
    std::vector<float> input_data = std::vector<float>(network_width * network_height * channels);
    std::memcpy(input_data.data(), input_blob.data, img_byte_size);

    std::vector<cv::Mat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::Mat(cv::Size(network_width, network_height), CV_32FC1, &(input_data[i * network_width * network_height])));
    }
    cv::split(input_blob, chw);

    return input_data;    
}