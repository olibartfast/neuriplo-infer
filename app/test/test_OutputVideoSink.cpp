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
class FakeWriter : public VideoWriterInterface {
public:
  bool initialize(const std::string &destination,
                  const videocapture::VideoWriterConfig &config) override {
    destination_ = destination;
    config_ = config;
    open_ = initialize_result_;
    return initialize_result_;
  }

  bool writeFrame(const videocapture::Frame &frame) override {
    copied_frames_ += 1;
    return recordFrame(frame);
  }

  bool writeFrame(videocapture::Frame &&frame) override {
    moved_frames_ += 1;
    return recordFrame(frame);
  }

  [[nodiscard]] bool isOpen() const override { return open_; }

  bool release() override {
    releases_ += 1;
    open_ = false;
    return release_result_;
  }

  bool initialize_result_ = true;
  bool write_result_ = true;
  bool release_result_ = true;

  std::string destination_;
  videocapture::VideoWriterConfig config_;
  std::vector<std::size_t> frame_sizes_;
  int copied_frames_ = 0;
  int moved_frames_ = 0;
  int releases_ = 0;
  bool open_ = false;

private:
  bool recordFrame(const videocapture::Frame &frame) {
    frame_sizes_.push_back(frame.storageSizeBytes());
    return write_result_;
  }
};

videocapture::Frame makeFrame(int width = 4, int height = 4) {
  return videocapture::Frame(width, height, videocapture::PixelFormat::BGR8);
}

std::filesystem::path destinationFor(const std::string &name) {
  return std::filesystem::temp_directory_path() /
         ("neuriplo_infer_sink_" + std::to_string(::getpid()) + "_" + name);
}

} // namespace

TEST(OutputVideoSinkTest, FinishReportsADestinationThatCouldNotBeCompleted) {
  auto writer = std::make_unique<FakeWriter>();
  writer->release_result_ = false;
  auto *raw = writer.get();
  const auto destination = destinationFor("failed_finalize.avi");

  neuriplo_infer::OutputVideoSink sink(destination.string(), 4, 4,
                                       std::move(writer));
  sink.write(makeFrame(), 0);

  // The frame was accepted; the encoder failed behind it, which only release()
  // can report. A run that ignored this would exit 0 over a truncated file.
  EXPECT_THROW(sink.finish(), std::runtime_error);
  EXPECT_EQ(raw->releases_, 1);
}

TEST(OutputVideoSinkTest, FinishNamesTheDestinationItCouldNotComplete) {
  auto writer = std::make_unique<FakeWriter>();
  writer->release_result_ = false;
  const auto destination = destinationFor("named_destination.avi");

  neuriplo_infer::OutputVideoSink sink(destination.string(), 4, 4,
                                       std::move(writer));
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
  auto writer = std::make_unique<FakeWriter>();
  auto *raw = writer.get();
  {
    neuriplo_infer::OutputVideoSink sink(
        destinationFor("released_once.avi").string(), 4, 4, std::move(writer));
    sink.finish();
    sink.finish(); // Idempotent: a second close is not a second finalize.
    EXPECT_EQ(raw->releases_, 1);
  }
  EXPECT_EQ(raw->releases_, 1);
}

TEST(OutputVideoSinkTest, TheDestructorStillFinalizesAnEarlyExit) {
  auto writer = std::make_unique<FakeWriter>();
  auto *raw = writer.get();
  {
    // The q/Escape path: the sink is destroyed without finish() ever running.
    neuriplo_infer::OutputVideoSink sink(
        destinationFor("early_exit.avi").string(), 4, 4, std::move(writer));
    sink.write(makeFrame(), 0);
  }
  EXPECT_EQ(raw->releases_, 1);
}

TEST(OutputVideoSinkTest, TheDestructorDoesNotThrowOnAFailedFinalize) {
  auto writer = std::make_unique<FakeWriter>();
  writer->release_result_ = false;
  auto *raw = writer.get();
  // Throwing here would run during unwinding and terminate the process, so the
  // destructor logs instead. The failure is reported by finish() on the paths
  // that reach it.
  EXPECT_NO_THROW({
    neuriplo_infer::OutputVideoSink sink(
        destinationFor("failed_finalize_dtor.avi").string(), 4, 4,
        std::move(writer));
    sink.write(makeFrame(), 0);
  });
  EXPECT_EQ(raw->releases_, 1);
}

TEST(OutputVideoSinkTest, FramesAreHandedOverByMoveNotCopied) {
  auto writer = std::make_unique<FakeWriter>();
  auto *raw = writer.get();
  neuriplo_infer::OutputVideoSink sink(
      destinationFor("moved_frames.avi").string(), 4, 4, std::move(writer));

  sink.write(makeFrame(), 0);
  sink.write(makeFrame(), 1);

  // The rvalue overload is what lets the encoder thread take the pixels
  // instead of copying them; a lvalue hand-off would still compile and still
  // write the file, so it is asserted rather than assumed.
  EXPECT_EQ(raw->moved_frames_, 2);
  EXPECT_EQ(raw->copied_frames_, 0);
  sink.finish();
}

TEST(OutputVideoSinkTest, ARejectedFrameFailsTheRunNamingItsIndex) {
  auto writer = std::make_unique<FakeWriter>();
  writer->write_result_ = false;
  neuriplo_infer::OutputVideoSink sink(
      destinationFor("rejected_frame.avi").string(), 4, 4, std::move(writer));

  try {
    sink.write(makeFrame(), 7);
    FAIL() << "a rejected frame was treated as skippable";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("frame 7"), std::string::npos);
  }
}

TEST(OutputVideoSinkTest, AWriterThatCannotOpenItsDestinationFailsImmediately) {
  auto writer = std::make_unique<FakeWriter>();
  writer->initialize_result_ = false;

  EXPECT_THROW(
      neuriplo_infer::OutputVideoSink(destinationFor("unopenable.avi").string(),
                                      4, 4, std::move(writer)),
      std::runtime_error);
}

TEST(OutputVideoSinkTest, TheDestinationsParentDirectoryIsCreated) {
  auto writer = std::make_unique<FakeWriter>();
  auto *raw = writer.get();
  const auto destination =
      destinationFor("nested_parent") / "deeper" / "annotated.avi";
  std::filesystem::remove_all(destination.parent_path().parent_path());

  neuriplo_infer::OutputVideoSink sink(destination.string(), 8, 6,
                                       std::move(writer));

  EXPECT_TRUE(std::filesystem::is_directory(destination.parent_path()));
  EXPECT_EQ(raw->destination_, destination.string());
  EXPECT_EQ(raw->config_.width, 8);
  EXPECT_EQ(raw->config_.height, 6);
  sink.finish();
  std::filesystem::remove_all(destination.parent_path().parent_path());
}

#endif // VIDEOCAPTURE_WITH_WRITER
