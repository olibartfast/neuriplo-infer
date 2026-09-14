# Feature Validation — Async video writer

> Written before implementation, so the criteria cannot be fitted to whatever
> gets built. Execute from this file at Group 7 — do not summarize it from
> memory.

The central claim is falsifiable and must be tested as such: **a writer-enabled
build encodes annotated video on a worker thread behind a bounded queue, still
writes every source frame exactly once, fails the run when the writer fails, and
finalizes the file on every exit path.** The default build is unchanged.

Requirement → check traceability: R-1 async stage, R-2 order/no-drop, R-3
failure propagation, R-4 flush/join on every exit, R-5 queue-wait metric, R-6
test seam, R-7 default build unchanged.

## The scoreboard

Run these three, in order, and record the output:

```bash
# 1. Default build (writer OFF) — must configure and build unchanged.
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 2. Default test build + full suite.
cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-test
ctest --test-dir build-test --output-on-failure

# 3. Writer-enabled test build.
cmake -S . -B build-test-writer -DDEFAULT_BACKEND=OPENCV_DNN \
  -DENABLE_APP_TESTS=ON -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-test-writer
ctest --test-dir build-test-writer --output-on-failure
```

## Automated

- [x] **R-1 (async stage).** A writer-enabled run with a `FakeVideoWriter` that
      blocks each frame completes and every `writeFrame` call happens on a
      thread other than the frame loop's. The backpressure test asserts
      `probe->worker_thread != std::this_thread::get_id()`.
- [x] **R-2 (order, no drop).** `AVideoRunWritesEveryFrameToTheOutputVideo` and
      `AClassificationVideoWritesEachSourceFrameExactlyOnce` pass unchanged. The
      new `AFullWriterQueueAppliesBackpressureWithoutDroppingFrames` uses a slow
      fake writer and a 12-frame source (deeper than the queue) and asserts 12
      writes with strictly increasing first-byte values.
- [x] **R-3 (failure propagation).** `AWriterFailureAfterNFramesFailsTheRunAtRender`:
      the fake writer returns false on write 3; the run throws, the report's
      failure stage is `render`, and `samples` is 0.
- [x] **R-4 (flush/join every exit).** `AsyncVideoWriterSinkTest.DestructorFinalizesACompleteFile`
      writes three frames, destroys the sink without `finish()`, and re-opens the
      destination: three readable frames. The exception path is covered by R-3
      (the run throws and the sink is destroyed during unwinding).
- [x] **R-5 (queue-wait metric).** `WriterQueueWaitIsReportedWhenTheWriterBlocks`
      reports a non-null, positive `metrics.writer_queue_wait_ms`;
      `WriterQueueWaitStaysNullWithoutAnOutputVideo` (both builds) and
      `RunReportFile.WriterQueueWaitIsSeparateAndNullUntilMeasured` cover the
      null case.
- [x] **R-6 (test seam).** The fake writer is injected through
      `InferencePipeline::video_writer_factory`; production leaves it empty and
      `AsyncVideoWriterSink` uses `createVideoWriter()`.
- [x] **R-7 (default build unchanged).** `--capabilities` from the default
      binary is byte-identical to the one built from `develop` (16831 bytes,
      `diff` clean); `capabilities_cli_contract`,
      `capabilities_schema_contract`, and `no_orphan_sources` pass in both
      builds; `RunReport::kSchemaVersion` stays 1.
- [x] The full suites pass: **179/179** default (`build-test`) and **190/190**
      writer (`build-test-writer`).
- [x] `python3 scripts/sync_supported_model_types.py --check
      --neuriplo-tasks-readme /home/oli/repos/neuriplo-tasks/README.md`:
      `check passed — all files are up to date`.
- [x] `clang-format-18 --dry-run --Werror` on every changed file is clean;
      `cppcheck --enable=warning --std=c++20 --error-exitcode=1 …` exits 0;
      `clang-tidy-18 -p build-test-writer` introduces no new warning
      (`processVideo`'s cognitive complexity is back under threshold after the
      finalizer was factored out; only pre-existing findings remain).
  > Deviation: `pre-commit` is not installed in this environment
  > (`command not found`), so the pre-commit hook wrapper did not run. The
  > underlying clang-format, cppcheck, and clang-tidy checks above were run
  > directly with the same tools and options as `lint.yml`.

## Manual

- [ ] Re-run the issue's measurement (end-to-end fps with and without
      `--output_video`) if weights and a sample video are present; otherwise
      record the deviation, as the predecessor packet did.
  > Deviation (2026-09-14): no model weights or sample video are present in this
  > environment, so the end-to-end fps comparison was not executed. The
  > predecessor packet recorded the same deviation. The async behavior is
  > covered by the automated tests above.
- [ ] A run stopped early with `q`/Escape still produces a playable file
      (shares the R-4 destructor path).
  > Deviation: a headless test cannot drive `cv::waitKey` to return `q`/Escape.
  > The early-exit path is the sink destructor, which
  > `DestructorFinalizesACompleteFile` exercises directly.
- [x] Every hyperlink added to docs resolves.

## Definition of Done

- [x] Every requirement is implemented or explicitly deferred in
      `requirements.md`.
- [x] Nothing in *Out of Scope* was implemented anyway — in particular, no
      drop flag, no queue-depth flag, no new stage, no `schema_version` bump.
- [x] The default build's `--capabilities` output is byte-identical.
- [x] Deviations from this file are recorded here, honestly, with what was run
      instead.
- [x] `CHANGELOG.md` and `docs/Usage.md` tell the same story as the code; the
      feature was not a roadmap phase, so `specs/roadmap.md` needs no status
      change.
