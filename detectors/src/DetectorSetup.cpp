#include "DetectorSetup.hpp"
#include "DFine.hpp"

std::unique_ptr<Detector> DetectorSetup::createDetector(const std::string& detectorType, const ModelInfo& model_info) {
    static const auto detectorCreators = getDetectorCreators(model_info);

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

std::unordered_map<std::string, std::function<std::unique_ptr<Detector>()>> DetectorSetup::getDetectorCreators(const ModelInfo& model_info) {
    return {
        // {"yolov4", [] { return std::make_unique<YoloV4>(); }},
        // {"yolov5", [] { return std::make_unique<YoloVn>(); }},
        // {"yolov6", [] { return std::make_unique<YoloVn>(); }},
        // {"yolov7", [] { return std::make_unique<YoloVn>(); }},
        // {"yolov8", [] { return std::make_unique<YoloVn>(); }},
        // {"yolov9", [] { return std::make_unique<YoloVn>(); }},
        // {"yolo11", [] { return std::make_unique<YoloVn>(); }},
        // {"yolonas", [] { return std::make_unique<YoloNas>(); }},
        // {"yolov10", [] { return std::make_unique<YOLOv10>(); }},
        // {"rtdetrul", [] { return std::make_unique<RtDetrUltralytics>(); }},
        // {"rtdetr", [] { return std::make_unique<RtDetr>(); }},
        {"dfine", [model_info] { return std::make_unique<DFine>(model_info); }}
    };
}