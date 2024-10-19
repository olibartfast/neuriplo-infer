#pragma once
#include "common.hpp"
#include "Detector.hpp"

class DetectorSetup {
public:
    static std::unique_ptr<Detector> createDetector(const std::string& detectorType);
};