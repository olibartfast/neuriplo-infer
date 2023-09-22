#pragma once
#include "common.hpp"
// Define a global logger variable
std::shared_ptr<spdlog::logger> logger;

void initializeLogger() {

    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    sinks.push_back( std::make_shared<spdlog::sinks::rotating_file_sink_mt>("output.log", 1024*1024*10, 3, true));
    logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));

    spdlog::register_logger(logger);
    logger->flush_on(spdlog::level::info);
}