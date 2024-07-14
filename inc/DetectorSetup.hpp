#pragma once
#include "common.hpp"
#include "YoloV4.hpp"
#include "YoloVn.hpp"
#include "YOLOv10.hpp"
#include "YoloNas.hpp"
#include "RtDetr.hpp"
#include "RtDetrUltralytics.hpp"


std::unique_ptr<Detector> createDetector(
    const std::string& detectorType);