# Feature Plan — async video writer (videocapture v0.6.0)

Derived from [`requirements.md`](requirements.md). Groups are in dependency
order; each ends in something observable.

## Group 1 — Move the pin

1. **Pin `videocapture` to `v0.6.0`** in `versions.env`
   (`VIDEOCAPTURE_VERSION=v0.6.0`, was `v0.5.0`). The tag is published upstream
   at `ebc7208`.
2. **Configure and build the writer variant** so the pin is proven before any
   app change is written against the new API:

   ```
   cmake -S . -B build-writer -DDEFAULT_BACKEND=OPENCV_DNN \
         -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build-writer
   ```

   Expected: the fetched `VideoCapture` reports `v0.6.0`, and the build fails
   nowhere except — possibly — on the ignored `release()` result, which
   Group 2 fixes. (`release()` becoming `bool` is source-compatible for a
   caller that discards it, so a clean build here is the likely outcome.)

## Group 2 — Behavior: report a destination that was not completed

3. **Give `OutputVideoSink` an explicit close.** Add
   `void finish()`, modelled on `FrameTimingsCsv::finish()`:
   - call `writer_->release()` once, guarded so a second call is a no-op;
   - throw `std::runtime_error` naming `--output_video` and the destination
     when it returns `false`, so a file that could not be encoded or finalized
     fails the run instead of leaving a truncated artifact behind an exit
     code of `0`.
   The destructor keeps releasing whatever `finish()` did not — the `q`/Escape
   and exception paths — and logs the failure rather than throwing, because a
   throwing destructor during unwinding terminates the process. The sink keeps
   the destination string for both messages.
4. **Close through it at both call sites.** In `processVideo` and
   `processVideoClassification`, replace `output_sink.reset()` with
   `output_sink->finish()` followed by the reset, in the same position — after
   `timings.finish()` and the capture `release()`, before the sample is
   counted, inside the existing `Render` attribution.
5. **Hand frames over by move.** `OutputVideoSink::write()` already takes its
   `videocapture::Frame` by value; forward it with `std::move` into
   `writeFrame(videocapture::Frame&&)`, the overload v0.6.0 adds so the encoder
   thread takes the pixels instead of copying them. Call sites are unchanged —
   `toFrame()` already returns an owning `Frame`.

## Group 3 — Contract and documentation

6. **Correct the comments that describe synchronous encoding.** The
   `OutputVideoSink::write()` comment block carries the measured per-frame cost
   and "Moving it behind a bounded queue is tracked in issue #49"; encoding now
   happens on the writer's thread, and what remains on the frame loop is the
   `toFrame()` copy plus the hand-off, with `writeFrame()` waiting only when the
   encoder falls behind. Rewrite it to say that, including where the wait is
   attributed (the render stage).
7. **`CHANGELOG.md` `[Unreleased]`:** a `Changed` entry for the pin move and
   what it buys (encoding off the frame loop, order preserved, no frames
   dropped), and a `Fixed` entry for the finalization failure now failing the
   run. Record that this resolves two of the `[0.10.0]` known limitations —
   the synchronous-encode entry and the unreported failed finalize — without
   rewriting that released section. Note the deferred queue-wait figure.
8. **`docs/Usage.md`:** the `--output_video` row states the fixed 30 fps and
   the build flag; add that encoding runs off the frame loop and that a
   destination that cannot be completed fails the run. **`docs/Deployment.md`:**
   note on the writer build flag that it needs a C++20 standard library
   providing `std::jthread` and `std::osyncstream` (the reason upstream's
   packaged macOS builds are capture-only), against the row that already lists
   macOS as untested.
9. **`specs/roadmap.md`:** add the phase this packet answers and record the
   deferred run-report queue-wait figure as its follow-up.

No `--capabilities`, schema, or generated-docs regeneration: no flag, value
type, or model type moves.

## Group 4 — Verification

10. **Tests.**
    - `app/test/test_FrameConversion.cpp`: the round-trip test discards
      `writer->release()`; assert it, since the re-read that follows only means
      something if the container was finalized.
    - `app/test/test_RunDiagnostics.cpp`: add a test that a destination the
      writer cannot complete fails the run — a path the writer accepts at
      `initialize()` but cannot finalize — and assert no sample is counted,
      mirroring `AVideoWithNoFramesFailsInsteadOfSkippingTheOutput`. If no
      such destination can be produced deterministically through the OpenCV
      writer on a plain runner, record that in `validation.md` and cover the
      propagation at the seam instead of inventing a flaky fixture.
    - The existing output-video tests (frame counts, every source frame exactly
      once, missing parent directories, the short-clip classification case)
      must pass unchanged — they are the ordering and completeness evidence.
11. **Run [`validation.md`](validation.md)** from the file and record evidence
    with dates.

Commit by group: the pin move and the behavior change are separately
reviewable, documentation and tests follow.
