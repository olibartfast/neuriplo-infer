#pragma once

#ifdef VIDEOCAPTURE_WITH_WRITER

#include "RunReport.hpp"
#include "VideoWriterInterface.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace neuriplo_infer {

/**
 * Encodes rendered frames on a dedicated worker thread behind a bounded FIFO
 * queue.
 *
 * The synchronous writer blocked the frame loop for the whole encode, which
 * capped end-to-end throughput well below what the inference backends reach.
 * Here the frame loop only hands off an owned frame; it blocks only when the
 * queue is full and the writer genuinely cannot keep up, and that wait is
 * reported separately in the run report.
 *
 * Every exit path is safe: the destructor stops the worker (draining queued
 * frames), joins it, and releases the writer, so the destination is a complete
 * playable file even when the run stops early or unwinds on an exception.
 * finish() is the explicit success path and rethrows the first worker failure
 * so a truncated output fails the run instead of being reported as success.
 */
class AsyncVideoWriterSink {
public:
  using WriterFactory = std::function<std::unique_ptr<VideoWriterInterface>()>;

  AsyncVideoWriterSink(std::string destination, int width, int height,
                       const WriterFactory &writer_factory, RunReport *report);
  ~AsyncVideoWriterSink();

  AsyncVideoWriterSink(const AsyncVideoWriterSink &) = delete;
  AsyncVideoWriterSink &operator=(const AsyncVideoWriterSink &) = delete;

  void write(videocapture::Frame frame, std::size_t frame_index);
  void finish();

private:
  struct QueuedFrame {
    videocapture::Frame frame;
    std::size_t frame_index{0};
  };

  void drain();
  void stop();

  std::string destination_;
  std::unique_ptr<VideoWriterInterface> writer_;
  RunReport *report_;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable space_available_;
  std::condition_variable frame_available_;
  std::deque<QueuedFrame> queue_;
  bool stopping_{false};
  bool joined_{false};
  bool failed_{false};
  std::exception_ptr error_;
};

} // namespace neuriplo_infer

#endif // VIDEOCAPTURE_WITH_WRITER
