#pragma once

#ifdef VIDEOCAPTURE_WITH_WRITER

#include "VideoWriterInterface.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace neuriplo_infer_test {

/// State shared between the test and the writer the sink owns. Counters are
/// atomic because writeFrame runs on the sink's worker thread.
struct WriterProbe {
  std::atomic<int> writes{0};
  std::atomic<bool> initialized{false};
  std::atomic<bool> released{false};
  /// 1-based writeFrame call that returns false; -1 never fails.
  int fail_on_write{-1};
  /// 1-based writeFrame call that throws; -1 never throws.
  int throw_on_write{-1};
  /// Slows each write so the producer outruns the queue and blocks.
  std::chrono::milliseconds per_frame_delay{0};
  /// When set before the sink starts, every writeFrame waits for openGate(),
  /// so a test can fill the queue deterministically instead of racing a sleep.
  bool gated{false};
  /// Thread that called writeFrame, so a test can prove encoding happens off
  /// the frame loop's thread.
  std::thread::id worker_thread;
  /// First byte of each written frame, in write order. Written only by the
  /// worker; read after join() establishes the happens-before edge.
  std::vector<int> first_bytes;

  void openGate() {
    {
      std::lock_guard<std::mutex> lock(gate_mutex);
      gate_open = true;
    }
    gate_cv.notify_all();
  }

  void waitForGate() {
    std::unique_lock<std::mutex> lock(gate_mutex);
    gate_cv.wait(lock, [this] { return gate_open; });
  }

  std::mutex gate_mutex;
  std::condition_variable gate_cv;
  bool gate_open{false};
};

class FakeVideoWriter : public VideoWriterInterface {
public:
  explicit FakeVideoWriter(std::shared_ptr<WriterProbe> probe)
      : probe_(std::move(probe)) {}

  bool initialize(const std::string & /*destination*/,
                  const videocapture::VideoWriterConfig & /*config*/) override {
    probe_->initialized = true;
    return true;
  }

  bool writeFrame(const videocapture::Frame &frame) override {
    const int call = ++probe_->writes;
    probe_->worker_thread = std::this_thread::get_id();
    if (probe_->gated) {
      probe_->waitForGate();
    }
    if (probe_->per_frame_delay.count() > 0) {
      std::this_thread::sleep_for(probe_->per_frame_delay);
    }
    if (call == probe_->throw_on_write) {
      throw std::runtime_error("fake writer: encoder threw");
    }
    const std::uint8_t *data = frame.data(0);
    if (data != nullptr) {
      probe_->first_bytes.push_back(static_cast<int>(data[0]));
    }
    return call != probe_->fail_on_write;
  }

  [[nodiscard]] bool isOpen() const override { return true; }

  void release() override { probe_->released = true; }

private:
  std::shared_ptr<WriterProbe> probe_;
};

} // namespace neuriplo_infer_test

#endif // VIDEOCAPTURE_WITH_WRITER
