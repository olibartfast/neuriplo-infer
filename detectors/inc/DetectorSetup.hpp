#pragma once
#include "Detector.hpp"



class DetectorSetup {
public:
    static std::unique_ptr<Detector> createDetector(const std::string& detectorType, const ModelInfo& model_info);

private:
    static std::unordered_map<std::string, std::function<std::unique_ptr<Detector>()>> getDetectorCreators(const ModelInfo& model_info);
};