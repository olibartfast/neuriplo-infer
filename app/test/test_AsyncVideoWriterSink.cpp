#include "AsyncVideoWriterSink.hpp"

#ifdef VIDEOCAPTURE_WITH_WRITER

#include "FakeVideoWriter.hpp"
#include "FrameConversion.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <string>

namespace {

class AsyncVideoWriterSinkTest : public ::testing::Test {
protected:
  void SetUp() override {
    directory_ =
        std::filesystem::temp_directory_path() / "neuriplo-async-writer-test";
    std::filesystem::remove_all(directory_);
    std::filesystem::create_directories(directory_);
  }

  void TearDown() override { std::filesystem::remove_all(directory_); }

  static int countFrames(const std::filesystem::path &path) {
    cv::VideoCapture capture(path.string());
    int frames = 0;
    cv::Mat frame;
    while (capture.read(frame)) {
      ++frames;
    }
    return frames;
  }

  std::filesystem::path directory_;
};

// The early-exit path: the frame loop stops (q/Escape) or unwinds without
// calling finish(). The destructor must drain the queue, join the worker, and
// release the writer, so the destination is a complete, playable file rather
// than a truncated one.
TEST_F(AsyncVideoWriterSinkTest, DestructorFinalizesACompleteFile) {
  const auto destination = directory_ / "annotated.avi";
  const cv::Mat image(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));
  {
    neuriplo_infer::AsyncVideoWriterSink sink(destination.string(), 64, 64, {},
                                              nullptr);
    sink.write(neuriplo_infer::toFrame(image), 0);
    sink.write(neuriplo_infer::toFrame(image), 1);
    sink.write(neuriplo_infer::toFrame(image), 2);
    // No finish(): destruction is the exit path under test.
  }

  EXPECT_EQ(countFrames(destination), 3);
}

// A worker failure must surface through the sink's API (from write() or
// finish()) instead of being swallowed and reporting a successful run with a
// truncated file.
TEST_F(AsyncVideoWriterSinkTest, WriterFailureSurfacesThroughTheSink) {
  const auto destination = directory_ / "annotated.avi";
  auto probe = std::make_shared<neuriplo_infer_test::WriterProbe>();
  probe->fail_on_write = 2;
  const cv::Mat image(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));

  neuriplo_infer::AsyncVideoWriterSink sink(
      destination.string(), 64, 64,
      [probe] {
        return std::make_unique<neuriplo_infer_test::FakeVideoWriter>(probe);
      },
      nullptr);

  bool threw = false;
  try {
    sink.write(neuriplo_infer::toFrame(image), 0);
    sink.write(neuriplo_infer::toFrame(image), 1);
    sink.write(neuriplo_infer::toFrame(image), 2);
    sink.finish();
  } catch (const std::runtime_error &) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

} // namespace

#endif // VIDEOCAPTURE_WITH_WRITER
