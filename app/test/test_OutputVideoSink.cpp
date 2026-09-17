#include <gtest/gtest.h>

#ifdef VIDEOCAPTURE_WITH_WRITER

#include "Frame.hpp"
#include "OutputVideoSink.hpp"
#include "VideoWriterConfig.hpp"
#include "VideoWriterInterface.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

// Drives the sink's failure paths. Since videocapture v0.6.0 the writer
// encodes on its own thread, so a frame that fails to encode is reported by a
// later writeFrame() or by release() -- and the OpenCV backend reports neither
// on demand: cv::VideoWriter returns nothing from write() or release(), and a
// write that runs out of disk is silently truncated. So the outcomes are set
// here rather than provoked through a codec.
//
// The sink owns the writer, and the destructor cases assert after the sink is
// gone, so what the test reads has to outlive the writer: the outcomes and the
// counters live in a state object both of them share.
struct FakeWriterState {
  bool initialize_result = true;
  bool write_result = true;
  bool release_result = true;

  std::string destination;
  videocapture::VideoWriterConfig config;
  std::vector<std::size_t> frame_sizes;
  int copied_frames = 0;
  int moved_frames = 0;
  int releases = 0;
  bool open = false;
};

class FakeWriter : public VideoWriterInterface {
public:
  explicit FakeWriter(std::shared_ptr<FakeWriterState> state)
      : state_(std::move(state)) {}

  bool initialize(const std::string &destination,
                  const videocapture::VideoWriterConfig &config) override {
    state_->destination = destination;
    state_->config = config;
    state_->open = state_->initialize_result;
    return state_->initialize_result;
  }

  bool writeFrame(const videocapture::Frame &frame) override {
    state_->copied_frames += 1;
    return recordFrame(frame);
  }

  bool writeFrame(videocapture::Frame &&frame) override {
    state_->moved_frames += 1;
    return recordFrame(frame);
  }

  [[nodiscard]] bool isOpen() const override { return state_->open; }

  bool release() override {
    state_->releases += 1;
    state_->open = false;
    return state_->release_result;
  }

private:
  bool recordFrame(const videocapture::Frame &frame) {
    state_->frame_sizes.push_back(frame.storageSizeBytes());
    return state_->write_result;
  }

  std::shared_ptr<FakeWriterState> state_;
};

std::unique_ptr<VideoWriterInterface>
fakeWriter(const std::shared_ptr<FakeWriterState> &state) {
  return std::make_unique<FakeWriter>(state);
}

videocapture::Frame makeFrame(int width = 4, int height = 4) {
  return videocapture::Frame(width, height, videocapture::PixelFormat::BGR8);
}

std::filesystem::path destinationFor(const std::string &name) {
  return std::filesystem::temp_directory_path() /
         ("neuriplo_infer_sink_" + std::to_string(::getpid()) + "_" + name);
}

} // namespace

TEST(OutputVideoSinkTest, FinishReportsADestinationThatCouldNotBeCompleted) {
  auto state = std::make_shared<FakeWriterState>();
  state->release_result = false;
  const auto destination = destinationFor("failed_finalize.avi");

  neuriplo_infer::OutputVideoSink sink(destination.string(), 4, 4,
                                       fakeWriter(state));
  sink.write(makeFrame(), 0);

  // The frame was accepted; the encoder failed behind it, which only release()
  // can report. A run that ignored this would exit 0 over a truncated file.
  EXPECT_THROW(sink.finish(), std::runtime_error);
  EXPECT_EQ(state->releases, 1);
}

TEST(OutputVideoSinkTest, FinishNamesTheDestinationItCouldNotComplete) {
  auto state = std::make_shared<FakeWriterState>();
  state->release_result = false;
  const auto destination = destinationFor("named_destination.avi");

  neuriplo_infer::OutputVideoSink sink(destination.string(), 4, 4,
                                       fakeWriter(state));
  try {
    sink.finish();
    FAIL() << "finish() accepted a destination the writer did not complete";
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("--output_video"), std::string::npos);
    EXPECT_NE(message.find(destination.string()), std::string::npos);
  }
}

TEST(OutputVideoSinkTest, FinishReleasesOnceAndTheDestructorDoesNotRepeatIt) {
  auto state = std::make_shared<FakeWriterState>();
  {
    neuriplo_infer::OutputVideoSink sink(
        destinationFor("released_once.avi").string(), 4, 4, fakeWriter(state));
    sink.finish();
    sink.finish(); // Idempotent: a second close is not a second finalize.
    EXPECT_EQ(state->releases, 1);
  }
  EXPECT_EQ(state->releases, 1);
}

TEST(OutputVideoSinkTest, TheDestructorStillFinalizesAnUnfinishedSink) {
  auto state = std::make_shared<FakeWriterState>();
  {
    // What unwinding does: the sink is destroyed without finish() ever running.
    neuriplo_infer::OutputVideoSink sink(
        destinationFor("unfinished.avi").string(), 4, 4, fakeWriter(state));
    sink.write(makeFrame(), 0);
  }
  EXPECT_EQ(state->releases, 1);
}

TEST(OutputVideoSinkTest, TheDestructorDoesNotThrowOnAFailedFinalize) {
  auto state = std::make_shared<FakeWriterState>();
  state->release_result = false;
  // Throwing here would run during unwinding and terminate the process, so the
  // destructor logs instead. The failure is reported by finish() on the paths
  // that reach it, which is every path that is not already failing.
  EXPECT_NO_THROW({
    neuriplo_infer::OutputVideoSink sink(
        destinationFor("failed_finalize_dtor.avi").string(), 4, 4,
        fakeWriter(state));
    sink.write(makeFrame(), 0);
  });
  EXPECT_EQ(state->releases, 1);
}

TEST(OutputVideoSinkTest, FramesAreHandedOverByMoveNotCopied) {
  auto state = std::make_shared<FakeWriterState>();
  neuriplo_infer::OutputVideoSink sink(
      destinationFor("moved_frames.avi").string(), 4, 4, fakeWriter(state));

  sink.write(makeFrame(), 0);
  sink.write(makeFrame(), 1);

  // The rvalue overload is what lets the encoder thread take the pixels
  // instead of copying them; an lvalue hand-off would still compile and still
  // write the file, so it is asserted rather than assumed.
  EXPECT_EQ(state->moved_frames, 2);
  EXPECT_EQ(state->copied_frames, 0);
  sink.finish();
}

TEST(OutputVideoSinkTest, ARejectedFrameFailsTheRunNamingItsIndex) {
  auto state = std::make_shared<FakeWriterState>();
  state->write_result = false;
  neuriplo_infer::OutputVideoSink sink(
      destinationFor("rejected_frame.avi").string(), 4, 4, fakeWriter(state));

  try {
    sink.write(makeFrame(), 7);
    FAIL() << "a rejected frame was treated as skippable";
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    // "at or before", because the encoder holds a few frames and reports a
    // failure through the first call after it: the index is where the run
    // noticed, and claiming it is the frame that failed would be wrong.
    EXPECT_NE(message.find("at or before frame 7"), std::string::npos);
  }
}

TEST(OutputVideoSinkTest, AWriterThatCannotOpenItsDestinationFailsImmediately) {
  auto state = std::make_shared<FakeWriterState>();
  state->initialize_result = false;

  EXPECT_THROW(
      neuriplo_infer::OutputVideoSink(destinationFor("unopenable.avi").string(),
                                      4, 4, fakeWriter(state)),
      std::runtime_error);
}

TEST(OutputVideoSinkTest, TheDestinationsParentDirectoryIsCreated) {
  auto state = std::make_shared<FakeWriterState>();
  const auto destination =
      destinationFor("nested_parent") / "deeper" / "annotated.avi";
  std::filesystem::remove_all(destination.parent_path().parent_path());

  neuriplo_infer::OutputVideoSink sink(destination.string(), 8, 6,
                                       fakeWriter(state));

  EXPECT_TRUE(std::filesystem::is_directory(destination.parent_path()));
  EXPECT_EQ(state->destination, destination.string());
  EXPECT_EQ(state->config.width, 8);
  EXPECT_EQ(state->config.height, 6);
  sink.finish();
  std::filesystem::remove_all(destination.parent_path().parent_path());
}

#endif // VIDEOCAPTURE_WITH_WRITER
