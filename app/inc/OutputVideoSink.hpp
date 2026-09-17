#pragma once

#ifdef VIDEOCAPTURE_WITH_WRITER

#include "VideoWriterInterface.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace neuriplo_infer {

// RAII over the videocapture writer behind `--output_video`: creates and
// initializes exactly once and guarantees release() on every exit path, so the
// destination is a complete playable file even when the run stops early.
//
// Since videocapture v0.6.0 the writer encodes on a thread of its own behind a
// bounded queue, so a frame that fails to encode is reported by a later
// writeFrame() or by release() rather than by the call that submitted it.
// finish() is what turns that last report into a failed run.
class OutputVideoSink {
public:
  // Writes through the backend videocapture selected for this build.
  OutputVideoSink(const std::string &destination, int width, int height);

  // Test seam. The OpenCV writer backend cannot be made to fail on demand --
  // cv::VideoWriter reports nothing from write() or release(), and a write
  // that hits a full disk is silently truncated -- so the failure paths are
  // driven through a writer supplied here instead of through a codec.
  OutputVideoSink(const std::string &destination, int width, int height,
                  std::unique_ptr<VideoWriterInterface> writer);

  // Last resort, for the paths finish() never reaches -- an exception on its
  // way out, a caller that forgot. It cannot report what it finds: throwing
  // here during unwinding terminates the process, and the failure already
  // travelling up is the one worth reporting. So it logs. Every path that ends
  // a run normally, including an operator stopping it early with q/Escape,
  // closes through finish() instead.
  ~OutputVideoSink();

  OutputVideoSink(const OutputVideoSink &) = delete;
  OutputVideoSink &operator=(const OutputVideoSink &) = delete;

  // writeFrame false is a failed run, not a skippable frame.
  //
  // This hands the frame to the encoder rather than encoding it: what the
  // frame loop pays is the cv::Mat -> Frame copy plus the hand-off, and the
  // wait when the encoder falls behind. Frames keep submission order and are
  // never dropped. Both the hand-off and any backpressure wait land in the
  // render stage, after the inference span, so per-inference latency is
  // unaffected.
  //
  // frame_index is where the run noticed a failure, not necessarily the frame
  // that failed: the encoder holds a few frames, and a failure is reported by
  // the first call after it happened. The message says so.
  void write(videocapture::Frame frame, std::size_t frame_index);

  // Encodes every accepted frame, finalizes the container, and reports the
  // outcome -- the counterpart of FrameTimingsCsv::finish(). A destination the
  // operator asked for and did not get is a failed run, not a warning, so this
  // throws rather than logging. Called on the normal path before the source is
  // counted as processed; the destructor covers the rest.
  void finish();

private:
  std::unique_ptr<VideoWriterInterface> writer_;
  std::string destination_;
  bool released_ = false;
};

} // namespace neuriplo_infer

#endif // VIDEOCAPTURE_WITH_WRITER
