#include "DetectorSetup.hpp"
#include "YoloV4.hpp"
#include "YoloVn.hpp"
#include "YOLOv10.hpp"
#include "YoloNas.hpp"
#include "RtDetr.hpp"
#include "RtDetrUltralytics.hpp"

std::unique_ptr<Detector> DetectorSetup::createDetector(const std::string& detectorType) {
    // Use a map to store the detector creators
    static const std::unordered_map<std::string, std::function<std::unique_ptr<Detector>()>> detectorCreators = {
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

    auto it = detectorCreators.find(detectorType);
    if (it != detectorCreators.end()) {
        return it->second();
    }
    return nullptr;
}