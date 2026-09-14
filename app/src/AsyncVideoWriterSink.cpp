#include "AsyncVideoWriterSink.hpp"

#ifdef VIDEOCAPTURE_WITH_WRITER

#include "VideoWriterConfig.hpp"
#include "VideoWriterFactory.hpp"

#include <glog/logging.h>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace neuriplo_infer {
namespace {

// Frames in flight: enough to absorb encoder jitter, bounded so the queue
// cannot grow without limit when the writer is slower than inference. At 1440p
// BGR8 that is roughly 44 MB of owned frame storage.
constexpr std::size_t kQueueDepth = 4;

} // namespace

AsyncVideoWriterSink::AsyncVideoWriterSink(std::string destination, int width,
                                           int height,
                                           const WriterFactory &writer_factory,
                                           RunReport *report)
    : destination_(std::move(destination)), report_(report) {
  // Same as the timings CSV and the run report: a destination below a
  // directory that does not exist yet is created rather than refused.
  const std::filesystem::path destination_path(destination_);
  if (destination_path.has_parent_path()) {
    std::filesystem::create_directories(destination_path.parent_path());
  }
  videocapture::VideoWriterConfig config;
  config.width = width;
  config.height = height;
  config.frameRate = 30.0;
  config.codec = videocapture::VideoCodec::Auto;
  writer_ = writer_factory ? writer_factory() : createVideoWriter();
  if (!writer_) {
    throw std::runtime_error(
        "--output_video: no writer backend available for destination: " +
        destination_);
  }
  // Initialized on the frame loop's thread, before the worker starts, so an
  // unwritable destination fails the run at the first frame rather than
  // surfacing as a deferred worker error.
  if (!writer_->initialize(destination_, config)) {
    throw std::runtime_error(
        "--output_video: could not initialize the video writer for " +
        destination_);
  }
  // A video is being written from here on, so the metric is measured: a writer
  // that never fills the queue reports zero wait, not the null of a run with no
  // --output_video.
  if (report_ != nullptr) {
    report_->addWriterQueueWaitMs(0.0);
  }
  worker_ = std::thread([this] { drain(); });
}

AsyncVideoWriterSink::~AsyncVideoWriterSink() {
  // Never throws: joining is the one thing that must happen on every exit
  // path, and a worker failure already propagated through write()/finish().
  stop();
  // The destructor also runs while unwinding, where a second exception would
  // terminate the process before the run report records the first.
  try {
    writer_->release();
  } catch (const std::exception &e) {
    LOG(WARNING) << "--output_video: releasing the video writer failed: "
                 << e.what();
  } catch (...) {
    LOG(WARNING) << "--output_video: releasing the video writer failed";
  }
}

void AsyncVideoWriterSink::write(videocapture::Frame frame,
                                 std::size_t frame_index) {
  const auto wait_start = std::chrono::steady_clock::now();
  bool blocked = false;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    // Only a wait that actually happened is backpressure; taking the lock and
    // moving the frame on a fast writer must not read as queue wait.
    blocked = queue_.size() >= kQueueDepth;
    space_available_.wait(
        lock, [this] { return queue_.size() < kQueueDepth || failed_; });
    if (failed_) {
      std::rethrow_exception(error_);
    }
    queue_.push_back(QueuedFrame{std::move(frame), frame_index});
  }
  if (blocked && report_ != nullptr) {
    report_->addWriterQueueWaitMs(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start)
            .count());
  }
  frame_available_.notify_one();
}

void AsyncVideoWriterSink::finish() {
  stop();
  // join() established a happens-before edge, so the worker's state is visible
  // here without the mutex.
  if (failed_) {
    std::rethrow_exception(error_);
  }
}

void AsyncVideoWriterSink::drain() {
  while (true) {
    QueuedFrame queued;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      frame_available_.wait(lock,
                            [this] { return !queue_.empty() || stopping_; });
      if (queue_.empty()) {
        return; // Stopped and drained.
      }
      queued = std::move(queue_.front());
      queue_.pop_front();
    }
    space_available_.notify_one();
    std::exception_ptr error;
    try {
      if (!writer_->writeFrame(queued.frame)) {
        error = std::make_exception_ptr(std::runtime_error(
            "--output_video: video writer failed to write frame " +
            std::to_string(queued.frame_index)));
      }
    } catch (...) {
      // An exception escaping a std::thread terminates the process; carried
      // across as state it fails the run at render like a false return.
      error = std::current_exception();
    }
    if (error) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
        error_ = std::move(error);
        // Anything still queued cannot be written: the first failure ends the
        // run, and the frame loop rethrows it from its next write().
        queue_.clear();
        stopping_ = true;
      }
      space_available_.notify_all();
      return;
    }
  }
}

void AsyncVideoWriterSink::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (joined_) {
      return;
    }
    stopping_ = true;
  }
  frame_available_.notify_all();
  space_available_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  joined_ = true;
}

} // namespace neuriplo_infer

#endif // VIDEOCAPTURE_WITH_WRITER
