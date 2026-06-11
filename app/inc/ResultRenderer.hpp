#pragma once

#include "AppConfig.hpp"
#include "TaskRouting.hpp"
#include "neuriplo/tasks/core/result_types.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct RenderContext {
  neuriplo_tasks::TaskType task_type;
  const std::vector<std::string> &classes;
  float confidence_threshold;
};

class ResultRenderer {
public:
  virtual ~ResultRenderer() = default;
  virtual void render(const std::vector<neuriplo_tasks::Result> &results,
                      cv::Mat &image, const RenderContext &context) = 0;
};

class DefaultResultRenderer : public ResultRenderer {
public:
  void render(const std::vector<neuriplo_tasks::Result> &results,
              cv::Mat &image, const RenderContext &context) override;
};
