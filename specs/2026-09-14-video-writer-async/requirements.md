# Feature Requirements — Async video writer (`--output_video` off the inference thread)

Roadmap phase: follow-up to the shipped `--output_video` feature; tracked by
issue [#49](https://github.com/olibartfast/neuriplo-infer/issues/49) (not a
separate roadmap phase).
Branch: `feature/video-writer-async`
Predecessor packet: [`2026-09-05-video-writer-sink/`](../2026-09-05-video-writer-sink/requirements.md)

## Goal

`--output_video` encodes and writes every annotated frame synchronously on the
frame loop's thread. Measured through the same videocapture v0.5.0 writer the app
uses, that is ~10.6 / 22.8 / 38.8 ms per 720p / 1080p / 1440p `.mp4` frame
(`.avi`/MJPG roughly 1.5x), and up to ~98 ms for 1440p noise. The write is
outside the timed inference span, so per-inference latency and `--timings_csv`
are not skewed, but it bounds end-to-end throughput well below what the
TensorRT backends reach. This feature moves encoding to a dedicated writer stage
behind a bounded queue so the frame loop only blocks when the writer genuinely
cannot keep up, and makes that backpressure visible in the run report.

## In Scope

- An asynchronous writer stage: one worker thread, one bounded FIFO queue,
  shared by both video frame loops (`processVideo`,
  `processVideoClassification`).
- Frame order preserved; `write()` hands off an owned `videocapture::Frame`.
- Backpressure policy: **block** when the queue is full (default, never drops a
  frame). No drop policy is added (see Out of Scope).
- The first writer failure is propagated back to the frame loop, so the run
  fails at the `render` stage instead of reporting success with a truncated
  file.
- Flush and join on every exit path: normal end of source, early `q`/Escape,
  and exception unwinding.
- Writer queue-wait time reported separately in the run report
  (`metrics.writer_queue_wait_ms`), so the backpressure cost stays visible.
- An injectable writer factory on `InferencePipeline` (writer builds only) so
  the failure and backpressure behaviors are testable without a real encoder
  failure.

## Out of Scope

- **A frame-drop policy / `--output_video_drop_frames`.** The issue says a drop
  policy only "if explicitly requested"; blocking is the honest default and no
  request was made. Deferred until asked for.
- **A configurable queue depth / `--output_video_queue_depth`.** A fixed small
  depth is enough to decouple the stages; a tuning knob is not needed yet.
- **`--output_fps` / `--output_codec`.** Still deferred from the predecessor
  packet; this branch does not touch them.
- **Multiple writer threads / parallel encode.** The encoder is the bottleneck,
  not the queue; one worker preserves order without extra synchronization.
- **A new run-report schema version or stage.** The queue-wait value is an
  additive metric, not a new `stages_ms` stage (see Decision 2).
- **Changing `toFrame` or the frame bridge.** Still one owned copy per frame.
- **Advertising anything new in `--capabilities`.** No CLI surface changes.

## Decisions

1. **A dedicated `AsyncVideoWriterSink` component, owned by the frame loop.**
   It creates and initializes the writer on the frame loop's thread (so an
   unwritable destination still fails fast at the first frame, before the worker
   starts), then starts the worker. `write()` blocks for a free slot, records
   the wait, and enqueues an owned frame. The destructor stops the worker
   (draining queued frames) and `release()`s the writer, so every exit path
   finalizes the file. `finish()` joins and rethrows the first worker failure for
   the explicit success path.

2. **Queue-wait is a new `metrics.writer_queue_wait_ms`, not a new stage.**
   The run report has a fixed eight-stage vocabulary that `--capabilities`
   advertises and tests assert; a `writer` stage would change that vocabulary
   and force a `schema_version` bump. A metric alongside
   `throughput_per_second` is additive, keeps `schema_version` at 1, and is
   `null` when no writer ran — consistent with "absent is not zero". It is
   accumulated on the frame-loop thread only, so `RunReport` stays single-threaded.

3. **The queue is bounded and blocking.** Fixed depth (a small constant, four
   frames), `std::condition_variable` for "space available" and "frame
   available". A full queue blocks the producer until the worker frees a slot; a
   frame is never dropped. The wait is only counted when the producer actually
   blocked (queue full on entry), so a fast writer reports ~0 rather than the
   cost of taking the mutex.

4. **Failures cross the thread boundary as state, not exceptions.** The worker
   catches nothing (it is not allowed to throw into the join), records the first
   `std::exception_ptr`, clears the queue, and stops. The producer rethrows it
   from the next `write()` or from `finish()`, so the run fails at `render` with
   the writer's own message.

5. **A test seam, not a test-only global.** `InferencePipeline` gains a
   `std::function<std::unique_ptr<VideoWriterInterface>()>` writer factory
   (compiled only in writer builds, empty in production). Tests inject a fake
   writer; the frame loops fall back to `createVideoWriter()` when it is empty.

6. **The sink is a separate compiled component.** `AsyncVideoWriterSink` lives
   in `app/inc/AsyncVideoWriterSink.hpp` / `app/src/AsyncVideoWriterSink.cpp`,
   added to `APP_LIB_SOURCES`, so the destructor/finalization path is unit
   testable directly (the early-exit path cannot be driven through a real
   keypress in a headless test). Its contents are guarded by
   `VIDEOCAPTURE_WITH_WRITER`.

## Constraints and Context

- **Repo boundary respected.** Encoding still belongs to `videocapture`; this
  branch only changes *when* `writeFrame()` is called, not how.
- **Thread and exception safety.** The worker is joined before the sink is
  destroyed and before the writer is released; no exception escapes the worker
  thread; `RunReport` is touched only by the frame-loop thread.
- **Memory.** A queued frame is an owned `Frame`; depth four bounds the queue at
  roughly four frame buffers (about 44 MB at 1440p BGR8).
- **No behavior change for the default build.** The new source is guarded by
  `VIDEOCAPTURE_WITH_WRITER`; a writer-less build compiles an empty translation
  unit. The new metric is `null` there.
- **Existing output-video tests must pass unchanged.** Frame count and
  every-source-frame-exactly-once guarantees are the predecessor's contract and
  stay intact.
- **Contract surfaces touched.** `RunReport` gains one metric and one
  accumulator; `InferencePipeline` gains one writer-build-only member; one new
  `app/src/*.cpp` is added to `APP_LIB_SOURCES`. No CLI, capabilities, or
  schema-version change.

## Open Questions

1. **Queue depth.** Four is a judgement call balancing decoupling against memory.
   If a future measurement shows the writer starving, raise it — the constant is
   isolated in one place.
2. **Real end-to-end measurement.** The issue asks for end-to-end fps with and
   without `--output_video` re-run. This environment has no model weights or
   sample video, so the run may be recorded as a deviation (as the predecessor
   packet did) unless weights are present.
