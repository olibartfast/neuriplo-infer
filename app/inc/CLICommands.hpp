#pragma once

#include "InferencePipeline.hpp"

#include <cstdint>
#include <vector>

class CLICommand {
public:
  virtual ~CLICommand() = default;
  virtual int execute(InferencePipeline &pipeline) = 0;
};

// encoded_source: the source's original file bytes in encoded-image mode, so
// warmup and benchmark send the same payload as the real request. Empty for
// every other mode, where the decoded image is what gets sent.
class WarmupCommand : public CLICommand {
public:
  explicit WarmupCommand(cv::Mat image,
                         std::vector<uint8_t> encoded_source = {});
  int execute(InferencePipeline &pipeline) override;

private:
  cv::Mat image_;
  std::vector<uint8_t> encoded_source_;
};

class BenchmarkCommand : public CLICommand {
public:
  explicit BenchmarkCommand(cv::Mat image,
                            std::vector<uint8_t> encoded_source = {});
  int execute(InferencePipeline &pipeline) override;

private:
  cv::Mat image_;
  std::vector<uint8_t> encoded_source_;
};

class RunInferenceCommand : public CLICommand {
public:
  int execute(InferencePipeline &pipeline) override;
};

class ExportMetadataCommand : public CLICommand {
public:
  int execute(InferencePipeline &pipeline) override;
};
