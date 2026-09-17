# Feature Validation — async video writer (videocapture v0.6.0)

Written before implementation. Execute from this file; record evidence and
dates, including deviations.

Requirement ↔ check:

| ID | Requirement | Check |
|----|-------------|-------|
| R-1 | `videocapture` pinned to `v0.6.0` and the app builds against it | V-1, V-2 |
| R-2 | A destination that fails to encode or finalize fails the run | V-3, V-4 |
| R-3 | Frames reach the writer through the move overload | V-5 |
| R-4 | Ordering and completeness unchanged (no dropped or reordered frames) | V-6 |
| R-5 | The writer-less default build is untouched | V-7 |
| R-6 | Docs and changelog describe v0.6.0 behavior, not the synchronous writer | V-8, V-9 |

## Automated

- [ ] **V-1** Writer build configures against the new pin and reports it:
      `cmake -S . -B build-writer -DDEFAULT_BACKEND=OPENCV_DNN -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | grep "VideoCapture: v0.6.0"`
- [ ] **V-2** Writer build compiles: `cmake --build build-writer`
- [ ] **V-3** Test build with the writer passes in full:
      `cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-test && ctest --test-dir build-test --output-on-failure`
- [ ] **V-4** A run whose destination cannot be completed exits non-zero and
      counts no sample — the new `RunDiagnostics` test, failing before the
      change and passing after.
- [ ] **V-5** The frame hand-off resolves to the rvalue overload: the
      round-trip writer test passes, and `OutputVideoSink::write` passes its
      frame with `std::move` (read the diff; a copy would still compile and
      still pass, so this one is inspected, not inferred).
- [ ] **V-6** The existing output-video tests pass unchanged:
      `ctest --test-dir build-test --output-on-failure -R 'RunDiagnostics|FrameConversion'`
      — frame counts (5, 3, 6), every source frame exactly once, created parent
      directories, zero-frame clip still fails.
- [ ] **V-7** Default writer-less build still configures and builds:
      `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [ ] **V-8** `pre-commit run --all-files` is clean (clang-format, cppcheck).
- [ ] **V-9** Generated docs in sync: `python3 scripts/sync_supported_model_types.py --check`
- [ ] `--capabilities` contract test still passes
      (`ctest --test-dir build-test -R capabilities`) — no schema move expected.

## Manual

- [ ] **M-1** A real annotated video run writes a playable file:
      `./build-writer/app/neuriplo-infer --source=<clip> --no_display --output_video=/tmp/annotated.avi`
      then re-read it and confirm the frame count matches the source.
- [ ] **M-2** Early exit still finalizes: interrupt a run with `q`, confirm the
      destination re-reads as a complete file with the frames written so far.
- [ ] **M-3** Throughput moved: the same clip with and without
      `--output_video`, comparing wall-clock and the run report's render stage.
      Encoding no longer serializes with the loop, so the gap should narrow
      against the v0.5.0 figures recorded in issue #49. Record both numbers —
      this is the evidence #49 asks for.
- [ ] **M-4** Every hyperlink added to docs and specs resolves (`ls` for
      relative paths, `curl -sI` for absolute).

## Definition of Done

- [ ] Every requirement implemented or explicitly deferred in `requirements.md`.
- [ ] Nothing in *Out of Scope* implemented anyway — in particular no
      run-report queue-wait figure and no app-side writer thread.
- [ ] Deviations recorded below, honestly, with what was run instead.
- [ ] `CHANGELOG.md` describes the user-visible change; `../roadmap.md` phase
      status updated.
- [ ] Spec, code, changelog, and roadmap tell the same story in one branch.

## Evidence

> Filled in at stage 6, with dates and the commands actually run.
