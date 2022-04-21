#include "Detector.hpp"


class HogSvmDetector : public Detector
{
    enum Mode { Default, Daimler } m;
    cv::HOGDescriptor hog, hog_d;
public:
    HogSvmDetector() : m(Default), hog(), hog_d(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9)
    {
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        hog_d.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
    }    
    std::string modeName() const;
    std::vector<cv::Rect> detect(cv::InputArray img);
    void adjustRect(cv::Rect & r) const;
    std::vector<Detection> run_detection(const cv::Mat& frame) override;
    ~HogSvmDetector(){};

};
