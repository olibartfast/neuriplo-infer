#include "DetectorSetup.hpp"
#include "YoloV4.hpp"
#include "YoloVn.hpp"
#include "YOLOv10.hpp"
#include "YoloNas.hpp"
#include "RtDetr.hpp"
#include "RtDetrUltralytics.hpp"

std::unique_ptr<Detector> DetectorSetup::createDetector(const std::string& detectorType) {
    static const auto detectorCreators = getDetectorCreators();

    auto it = detectorCreators.find(detectorType);
    if (it != detectorCreators.end()) {
        return it->second();
    } else {
        LOG(ERROR) << "Unknown detector type '" << detectorType << "' requested. Available types are: ";
        for (const auto& pair : detectorCreators) {
            LOG(ERROR) << pair.first;
        }
        throw std::invalid_argument("Unknown detector type");
    }
}

std::unordered_map<std::string, std::function<std::unique_ptr<Detector>()>> DetectorSetup::getDetectorCreators() {
    return {
        {"yolov4", [] { return std::make_unique<YoloV4>(); }},
        {"yolov5", [] { return std::make_unique<YoloVn>(); }},
        {"yolov6", [] { return std::make_unique<YoloVn>(); }},
        {"yolov7", [] { return std::make_unique<YoloVn>(); }},
        {"yolov8", [] { return std::make_unique<YoloVn>(); }},
        {"yolov9", [] { return std::make_unique<YoloVn>(); }},
        {"yolo11", [] { return std::make_unique<YoloVn>(); }},
        {"yolonas", [] { return std::make_unique<YoloNas>(); }},
        {"yolov10", [] { return std::make_unique<YOLOv10>(); }},
        {"rtdetrul", [] { return std::make_unique<RtDetrUltralytics>(); }},
        {"rtdetr", [] { return std::make_unique<RtDetr>(); }}
    };
}