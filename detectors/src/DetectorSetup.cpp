#include "DetectorSetup.hpp"
#include "YoloV4.hpp"
#include "YoloVn.hpp"
#include "YOLOv10.hpp"
#include "YoloNas.hpp"
#include "RtDetr.hpp"
#include "RtDetrUltralytics.hpp"
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
        {"yolov4", [model_info] { return std::make_unique<YoloV4>(model_info); }},
        {"yolov5", [model_info] { return std::make_unique<YoloVn>(model_info); }},
        {"yolov6", [model_info] { return std::make_unique<YoloVn>(model_info); }},
        {"yolov7", [model_info] { return std::make_unique<YoloVn>(model_info); }},
        {"yolov8", [model_info] { return std::make_unique<YoloVn>(model_info); }},
        {"yolov9", [model_info] { return std::make_unique<YoloVn>(model_info); }},
        {"yolo11", [model_info] { return std::make_unique<YoloVn>(model_info); }},
        {"yolonas", [model_info] { return std::make_unique<YoloNas>(model_info); }},
        {"yolov10", [model_info] { return std::make_unique<YOLOv10>(model_info); }},
        {"rtdetrul", [model_info] { return std::make_unique<RtDetrUltralytics>(model_info); }},
        {"rtdetr", [model_info] { return std::make_unique<RtDetr>(model_info); }},
        {"dfine", [model_info] { return std::make_unique<DFine>(model_info); }}
    };
}