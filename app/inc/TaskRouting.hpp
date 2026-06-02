#pragma once

#include "vision-core/core/result_types.hpp"

#include <string>

vision_core::TaskType getTaskTypeForModel(const std::string &model_type);
