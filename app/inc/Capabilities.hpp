#pragma once

#include <iosfwd>
#include <nlohmann/json_fwd.hpp>

nlohmann::json buildCapabilities();
void printCapabilities(std::ostream &output);
