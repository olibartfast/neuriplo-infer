#pragma once

// App-local, dependency-free copy of neuriplo's InferenceMetadata, used when
// neuriplo-infer is built WITHOUT local backends (KServe-only). It is put on
// the include path only in that build mode, so no source #include changes are
// needed between modes. Keep the public shape identical to neuriplo's version.
// See docs/KserveRuntime.md.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct LayerInfo {
  std::string name;
  std::vector<int64_t> shape;
  size_t batch_size;
};

class InferenceMetadata {
public:
  void addInput(const std::string &name, const std::vector<int64_t> &shape,
                size_t batch_size) {
    inputs_.push_back({name, shape, batch_size});
  }
  void addOutput(const std::string &name, const std::vector<int64_t> &shape,
                 size_t batch_size) {
    outputs_.push_back({name, shape, batch_size});
  }
  const std::vector<LayerInfo> &getInputs() const { return inputs_; }
  const std::vector<LayerInfo> &getOutputs() const { return outputs_; }

private:
  std::vector<LayerInfo> inputs_;
  std::vector<LayerInfo> outputs_;
};
