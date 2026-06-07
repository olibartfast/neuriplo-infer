#pragma once

#include "neuriplo/tasks/core/result_types.hpp"

#include <string>

neuriplo_tasks::TaskType getTaskTypeForModel(const std::string &model_type);
