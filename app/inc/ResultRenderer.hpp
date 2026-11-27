#pragma once

#include "AppConfig.hpp"
#include "TaskRouting.hpp"
#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/vision/image.hpp"

#include <string>
#include <vector>

struct RenderContext {
  neuriplo_tasks::TaskType task_type;
  const std::vector<std::string> &classes;
  float confidence_threshold;
  int pair_source_width{0};
};

class ResultRenderer {
public:
  virtual ~ResultRenderer() = default;
  virtual void render(const std::vector<neuriplo_tasks::Result> &results,
                      neuriplo_tasks::vision::Image &image,
                      const RenderContext &context) = 0;
};

class DefaultResultRenderer : public ResultRenderer {
public:
  void render(const std::vector<neuriplo_tasks::Result> &results,
              neuriplo_tasks::vision::Image &image,
              const RenderContext &context) override;
};

// Human-readable name for a class id from the labels file, or the numeric id
// when no labels file covers it. Shared by the renderers and the CLI log
// paths so both report the same name.
std::string labelForClass(float class_id,
                          const std::vector<std::string> &classes);
