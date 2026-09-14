# Feature Plan — Async video writer

> Derived from [`requirements.md`](requirements.md). Phases are in dependency
> order; each ends with the tree building and its phase-local check green. The
> default build stays unchanged.

## Group 1 — Packet (orchestrator)

1. Create `specs/2026-09-14-video-writer-async/` with the three artifacts and
   branch `feature/video-writer-async` from `develop`.

## Group 2 — Report metric (no writer dependency)

2. **`RunReport`**: add `addWriterQueueWaitMs(double)` and an optional
   `writer_queue_wait_ms_` accumulator, serialized into `metrics` as
   `writer_queue_wait_ms` (`null` until a writer reports). Keep
   `kSchemaVersion` at 1 and the eight-stage vocabulary unchanged.

Phase-local validation:

```bash
cmake --build build-test
ctest --test-dir build-test -R RunReport --output-on-failure
```

## Group 3 — Async sink (writer builds)

3. **`app/inc/AsyncVideoWriterSink.hpp` / `app/src/AsyncVideoWriterSink.cpp`**,
   guarded by `VIDEOCAPTURE_WITH_WRITER`, added to `APP_LIB_SOURCES`:
   - constructor: create parent dirs, build `VideoWriterConfig` (width, height,
     30 fps, `VideoCodec::Auto`), create the writer via the injected factory or
     `createVideoWriter()`, `initialize()` (throw naming `--output_video` and
     the destination on failure), then start the worker;
   - `write(Frame, frame_index)`: block while the bounded queue is full, count
     the wait when it actually blocked, enqueue, notify; rethrow the first
     worker failure;
   - worker: pop, `writeFrame()`; on false record the first `exception_ptr`,
     clear the queue, stop;
   - `finish()`: stop/join, then rethrow the first failure;
   - destructor: stop/join (draining queued frames) then `release()`; never
     throws.

Phase-local validation:

```bash
cmake --build build-writer
```

## Group 4 — Wiring (writer builds)

4. **`InferencePipeline`**: add the writer-build-only
   `video_writer_factory` test seam (include `VideoWriterInterface.hpp` and
   `<functional>` under the macro).

5. **`CLICommands.cpp`**: delete the synchronous `OutputVideoSink`, include
   `AsyncVideoWriterSink.hpp`, construct it with
   `pipeline.video_writer_factory` and `pipeline.report` in both frame loops, and
   replace the final `output_sink.reset()` with `finish()` + `reset()` so a
   worker failure fails the run at `render` before the sample is counted.

Phase-local validation:

```bash
cmake --build build-writer && cmake --build build-test-writer
```

## Group 5 — Tests (writer builds)

6. **`test_RunDiagnostics.cpp`**:
   - a `FakeVideoWriter` with a shared probe (write count, fail-after-N,
     optional per-frame delay, released flag);
   - writer failure after N frames fails the run at `render` and counts no
     sample;
   - queue-full backpressure writes every source frame exactly once (no drops);
   - queue-wait is reported in the run report when the producer blocked.

7. **Direct sink test** (new `test_AsyncVideoWriterSink.cpp`, registered in
   `app/test/CMakeLists.txt`, guarded by the macro): write a few frames, destroy
   without `finish()`, and re-open the destination to prove the destructor
   flushed, joined, and finalized a playable file (the early-exit path).

## Group 6 — Contract and documentation

8. **Docs.** `docs/Usage.md`: add `writer_queue_wait_ms` to the run-report
   example and a bullet explaining it. `CHANGELOG.md` `[Unreleased]`: `Changed`
   entry (writer moved off the inference thread) and `Added`
   (`writer_queue_wait_ms`); remove the now-fixed known-limitation bullet from
   `[0.10.0]`? No — released sections are immutable; add the `Changed` entry to
   `[Unreleased]`. `specs/roadmap.md`: remove the *Candidates* entry for #49 once
   delivered (in this branch).

9. **`specs/2026-09-14-video-writer-async/`** updated with what was actually
   built and validated.

## Group 7 — Verification

10. Execute [`validation.md`](validation.md) from the file: the three-build
    scoreboard and the automated suite. Record deviations honestly.

## Review and merge

11. Merge per
    [`../procedures/merge-feature-branch.md`](../procedures/merge-feature-branch.md):
    sync with `develop`, re-run the scoreboard, open the PR
    `feature/video-writer-async` → `develop`, merge as a merge commit, delete
    the branch. No release tag; the changelog `[Unreleased]` entries ship with
    the next release.

> Commit by phase: the report metric, the sink, the wiring + tests, and the docs
> are reviewable separately.
