#include "OutputVideoSink.hpp"

#ifdef VIDEOCAPTURE_WITH_WRITER

#include "VideoWriterConfig.hpp"
#include "VideoWriterFactory.hpp"

#include <glog/logging.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace neuriplo_infer {

OutputVideoSink::OutputVideoSink(const std::string &destination, int width,
                                 int height)
    : OutputVideoSink(destination, width, height, createVideoWriter()) {}

OutputVideoSink::OutputVideoSink(const std::string &destination, int width,
                                 int height,
                                 std::unique_ptr<VideoWriterInterface> writer)
    : writer_(std::move(writer)), destination_(destination) {
  // Same as the timings CSV and the run report: a destination below a
  // directory that does not exist yet is created rather than refused.
  const std::filesystem::path destination_path(destination);
  if (destination_path.has_parent_path()) {
    std::filesystem::create_directories(destination_path.parent_path());
  }
  videocapture::VideoWriterConfig config;
  config.width = width;
  config.height = height;
  config.frameRate = 30.0;
  config.codec = videocapture::VideoCodec::Auto;
  if (!writer_) {
    throw std::runtime_error(
        "--output_video: no writer backend available for destination: " +
        destination);
  }
  if (!writer_->initialize(destination, config)) {
    throw std::runtime_error(
        "--output_video: could not initialize the video writer for " +
        destination);
  }
}

OutputVideoSink::~OutputVideoSink() {
  if (released_ || !writer_) {
    return;
  }
  if (!writer_->release()) {
    LOG(ERROR) << "--output_video: " << destination_
               << " was not completed: a frame failed to encode or the "
                  "container could not be finalized";
  }
}

void OutputVideoSink::write(videocapture::Frame frame,
                            std::size_t frame_index) {
  if (!writer_->writeFrame(std::move(frame))) {
    // "at or before": the encoder runs behind the loop and holds a few frames,
    // and a failure is only reported by the first call after it happened, so
    // this index is where the run noticed -- not necessarily the frame that
    // failed to encode.
    throw std::runtime_error(
        "--output_video: video writer failed at or before frame " +
        std::to_string(frame_index));
  }
}

void OutputVideoSink::finish() {
  if (released_) {
    return;
  }
  released_ = true;
  if (!writer_->release()) {
    throw std::runtime_error(
        "--output_video: could not complete " + destination_ +
        ": a frame failed to encode or the container could not be finalized");
  }
}

} // namespace neuriplo_infer

#endif // VIDEOCAPTURE_WITH_WRITER
