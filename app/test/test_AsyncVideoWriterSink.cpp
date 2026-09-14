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
#include <thread>

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

// A backend that throws from writeFrame (a cv::Exception, bad_alloc) must fail
// the run through the sink like a false return, not escape the worker thread
// and terminate the process before the run report is written.
TEST_F(AsyncVideoWriterSinkTest, WriterExceptionSurfacesThroughTheSink) {
  const auto destination = directory_ / "annotated.avi";
  auto probe = std::make_shared<neuriplo_infer_test::WriterProbe>();
  probe->throw_on_write = 2;
  const cv::Mat image(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));

  neuriplo_infer::AsyncVideoWriterSink sink(
      destination.string(), 64, 64,
      [probe] {
        return std::make_unique<neuriplo_infer_test::FakeVideoWriter>(probe);
      },
      nullptr);

  std::string message;
  try {
    sink.write(neuriplo_infer::toFrame(image), 0);
    sink.write(neuriplo_infer::toFrame(image), 1);
    sink.write(neuriplo_infer::toFrame(image), 2);
    sink.finish();
  } catch (const std::runtime_error &e) {
    message = e.what();
  }
  EXPECT_EQ(message, "fake writer: encoder threw");
}

// A writer that keeps up never blocks the frame loop, so its wait is a measured
// zero rather than the null of a run with no --output_video.
TEST_F(AsyncVideoWriterSinkTest, UnblockedWriterReportsZeroQueueWait) {
  const auto destination = directory_ / "annotated.avi";
  auto probe = std::make_shared<neuriplo_infer_test::WriterProbe>();
  const cv::Mat image(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));
  neuriplo_infer::RunReport report;

  neuriplo_infer::AsyncVideoWriterSink sink(
      destination.string(), 64, 64,
      [probe] {
        return std::make_unique<neuriplo_infer_test::FakeVideoWriter>(probe);
      },
      &report);
  sink.write(neuriplo_infer::toFrame(image), 0);
  sink.finish();

  const auto wait = report.toJson().at("metrics").at("writer_queue_wait_ms");
  ASSERT_FALSE(wait.is_null());
  EXPECT_EQ(wait.get<double>(), 0.0);
}

// Backpressure is counted only when the producer finds the queue full. The
// gated writer holds the first frame, so the queue fills deterministically and
// the next write must block until the gate opens.
TEST_F(AsyncVideoWriterSinkTest, FullQueueWaitIsReported) {
  const auto destination = directory_ / "annotated.avi";
  auto probe = std::make_shared<neuriplo_infer_test::WriterProbe>();
  probe->gated = true;
  const cv::Mat image(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));
  neuriplo_infer::RunReport report;

  neuriplo_infer::AsyncVideoWriterSink sink(
      destination.string(), 64, 64,
      [probe] {
        return std::make_unique<neuriplo_infer_test::FakeVideoWriter>(probe);
      },
      &report);

  // Frame 0 is taken by the worker, which then waits at the gate.
  sink.write(neuriplo_infer::toFrame(image), 0);
  while (probe->writes.load() < 1) {
    std::this_thread::yield();
  }
  // Four more fill the queue without blocking.
  for (std::size_t index = 1; index <= 4; ++index) {
    sink.write(neuriplo_infer::toFrame(image), index);
  }
  std::thread opener([probe] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    probe->openGate();
  });
  // The queue is full on entry, so this write blocks until the gate opens.
  sink.write(neuriplo_infer::toFrame(image), 5);
  opener.join();
  sink.finish();

  EXPECT_EQ(probe->writes.load(), 6);
  const auto wait = report.toJson().at("metrics").at("writer_queue_wait_ms");
  ASSERT_FALSE(wait.is_null());
  EXPECT_GT(wait.get<double>(), 0.0);
}

} // namespace

#endif // VIDEOCAPTURE_WITH_WRITER
