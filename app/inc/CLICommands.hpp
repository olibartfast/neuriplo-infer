#pragma once

#include "InferencePipeline.hpp"

class CLICommand {
public:
  virtual ~CLICommand() = default;
  virtual int execute(InferencePipeline &pipeline) = 0;
};

class WarmupCommand : public CLICommand {
public:
  explicit WarmupCommand(cv::Mat image);
  int execute(InferencePipeline &pipeline) override;

private:
  cv::Mat image_;
};

class BenchmarkCommand : public CLICommand {
public:
  explicit BenchmarkCommand(cv::Mat image);
  int execute(InferencePipeline &pipeline) override;

private:
  cv::Mat image_;
};

class RunInferenceCommand : public CLICommand {
public:
  int execute(InferencePipeline &pipeline) override;
};

class ExportMetadataCommand : public CLICommand {
public:
  int execute(InferencePipeline &pipeline) override;
};
