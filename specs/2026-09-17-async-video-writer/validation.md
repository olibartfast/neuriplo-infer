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

- [x] **V-1** Writer build configures against the new pin and reports it:
      `cmake -S . -B build-writer -DDEFAULT_BACKEND=OPENCV_DNN -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | grep "VideoCapture: v0.6.0"`
- [x] **V-2** Writer build compiles: `cmake --build build-writer`
- [x] **V-3** Test build with the writer passes in full:
      `cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-test && ctest --test-dir build-test --output-on-failure`
- [x] **V-4** A run whose destination cannot be completed exits non-zero and
      counts no sample — the new `RunDiagnostics` test, failing before the
      change and passing after. **Deviation:** covered by
      `OutputVideoSinkTest` through an injectable writer instead; the OpenCV
      backend cannot be made to fail. See Evidence.
- [x] **V-5** The frame hand-off resolves to the rvalue overload: the
      round-trip writer test passes, and `OutputVideoSink::write` passes its
      frame with `std::move` (read the diff; a copy would still compile and
      still pass, so this one is inspected, not inferred).
- [x] **V-6** The existing output-video tests pass unchanged:
      `ctest --test-dir build-test --output-on-failure -R 'RunDiagnostics|FrameConversion'`
      — frame counts (5, 3, 6), every source frame exactly once, created parent
      directories, zero-frame clip still fails.
- [x] **V-7** Default writer-less build still configures and builds:
      `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [x] **V-8** `pre-commit run --all-files` is clean (clang-format, cppcheck).
      **Deviation:** `pre-commit` is not installed here; its tools were run
      directly. See Evidence.
- [x] **V-9** Generated docs in sync: `python3 scripts/sync_supported_model_types.py --check`
      **Deviation:** run against the pinned tag's README, not the stale local
      `build/_deps` copy the default path reads. See Evidence.
- [x] `--capabilities` contract test still passes
      (`ctest --test-dir build-test -R capabilities`) — no schema move expected.

## Manual

- [x] **M-1** A real annotated video run writes a playable file:
      `./build-writer/app/neuriplo-infer --source=<clip> --no_display --output_video=/tmp/annotated.avi`
      then re-read it and confirm the frame count matches the source.
- [x] **M-2** Early exit still finalizes: interrupt a run with `q`, confirm the
      destination re-reads as a complete file with the frames written so far.
      **Deviation:** no interactive session available; covered by the two
      destructor tests. See Evidence.
- [x] **M-3** Throughput moved: the same clip with and without
      `--output_video`, comparing wall-clock and the run report's render stage.
      Encoding no longer serializes with the loop, so the gap should narrow
      against the v0.5.0 figures recorded in issue #49. Record both numbers —
      this is the evidence #49 asks for.
- [x] **M-4** Every hyperlink added to docs and specs resolves (`ls` for
      relative paths, `curl -sI` for absolute).

## Definition of Done

- [x] Every requirement implemented or explicitly deferred in `requirements.md`.
- [x] Nothing in *Out of Scope* implemented anyway — in particular no
      run-report queue-wait figure and no app-side writer thread.
- [x] Deviations recorded below, honestly, with what was run instead.
- [x] `CHANGELOG.md` describes the user-visible change; `../roadmap.md` phase
      status updated.
- [x] Spec, code, changelog, and roadmap tell the same story in one branch.

## Evidence

Run 2026-09-17 on Ubuntu, AMD Ryzen 7 5700U, OpenCV 4.10.0, GCC with
`-DCMAKE_BUILD_TYPE=Release`.

**Where the builds ran.** Not in the working tree: CMake prefers a sibling
checkout when `../neuriplo-tasks` exists, and this machine's sibling is 10
commits behind `origin/develop` and predates `neuriplo_tasks::isImageInputShape`,
so *any* local build of `develop` fails on `app/src/ModelInputTypes.cpp` with or
without this branch. Builds therefore ran in a detached worktree outside
`~/repos`, where every sibling resolves to its pinned tag — the configuration
CI uses. Reported to the maintainer; not this branch's defect to fix.

- **V-1** `cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON -DCMAKE_BUILD_TYPE=Release`
  → `-- VideoCapture: v0.6.0 [source: versions.env]`, `-- VideoCapture writer: ON`,
  `-- neuriplo-tasks: v0.8.2 [source: versions.env]`.
- **V-2** `cmake --build build-test -j16` → exit 0, no warnings.
- **V-3** `ctest --test-dir build-test --output-on-failure` → **193/193 passed**
  (184 before this branch; 9 new `OutputVideoSinkTest` cases), 23.1 s.
- **V-4** *Deviation, recorded honestly.* The end-to-end fixture this file asked
  for does not exist: the OpenCV backend cannot be made to fail on demand. A
  probe (`cv::VideoWriter` under `RLIMIT_FSIZE` 64 KiB, `SIGXFSZ` ignored, 300
  frames at 640x480 MJPG) wrote a file truncated at exactly 65536 bytes with
  `isOpened()` still true and `release()` reporting nothing — the failure is
  invisible to the backend, so no destination can produce it. Covered instead at
  the seam: `OutputVideoSink` gained a constructor taking the writer to drive,
  and `app/test/test_OutputVideoSink.cpp` exercises a writer whose `release()`
  fails. Verified as a real check, not a tautology: against a sink that discards
  `release()` and hands frames over as lvalues, 4 of the 9 cases fail
  (`FinishReportsADestinationThatCouldNotBeCompleted`,
  `FinishNamesTheDestinationItCouldNotComplete`,
  `FinishReleasesOnceAndTheDestructorDoesNotRepeatIt`,
  `FramesAreHandedOverByMoveNotCopied`); all 9 pass against the branch.
- **V-5** `FramesAreHandedOverByMoveNotCopied` asserts the fake writer counted
  2 rvalue hand-offs and 0 copies; the diff passes `std::move(frame)` into
  `writeFrame`.
- **V-6** `ctest -R 'RunDiagnostics|FrameConversion'` → all passed, including the
  five output-video cases (5 / 3 / 6 frames written, every source frame exactly
  once, created parent directories, zero-frame clip still fails).
- **V-7** `cmake -S . -B build-default -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release && cmake --build build-default`
  → exit 0, `-- VideoCapture writer: OFF`. `OutputVideoSink.cpp` compiles to an
  empty translation unit there.
- **V-8** *Deviation:* `pre-commit` is not installed on this machine. Ran the
  hooks' own tools instead: `clang-format --dry-run --Werror` on every changed
  C++ file (clean), `cppcheck --enable=warning --std=c++20 --error-exitcode=1
  -Iapp/inc` over the changed sources (clean), and a trailing-whitespace /
  final-newline sweep of the diff (clean).
- **V-9** *Deviation in the command, not the outcome:* the script's default path
  reads `build/_deps/neuriplo-tasks-src/README.md`, a stale August build
  directory in this checkout, which reports the page out of date. Against the
  pinned tag's README (`git show v0.8.2:README.md`),
  `python3 scripts/sync_supported_model_types.py --check --neuriplo-tasks-readme <that file>`
  → `check passed — all files are up to date`.
- **Capabilities** contract test passed as part of the 193.

### M-1 — a real annotated video run

`ffmpeg` fixture: 30 frames, 640x640, h264. Command run from the worktree:

```
./build-test/app/neuriplo-infer --source=clip.mp4 --type=vitpose \
  --weights=vitpose_base.onnx --input_sizes=3,256,192 --no_display \
  --output_video=annotated.avi
```

Exit 0. `ffprobe -count_frames` on the destination: `mjpeg,640,640,30` against
the source's `h264,640,640,30` — every frame written, nothing dropped or
duplicated. (Two other local ONNX models were tried first and were rejected by
OpenCV 4.10's importer before the writer was ever reached, which is the
already-recorded `OPENCV_DNN` limitation, not a writer fault.)

### M-2 — early exit

*Deviation:* `q`/Escape needs a preview window and an interactive session, which
this environment does not have (`--no_display` is what makes the run possible at
all). The path is covered by `TheDestructorStillFinalizesAnEarlyExit` and
`TheDestructorDoesNotThrowOnAFailedFinalize`, which destroy the sink without
calling `finish()` and assert the writer was still released exactly once.

### M-3 — where the throughput went

**App level,** same clip, two runs each, wall clock:

| Configuration | Run 1 | Run 2 |
|---|---|---|
| without `--output_video` | 10.07 s | 10.02 s |
| with `--output_video` | 10.03 s | 9.86 s |

Writing the annotated video now costs nothing measurable end to end. At 640x640
the encode was small next to ViTPose inference (~330 ms/frame) even before, so
this run confirms the cost is gone rather than quantifying the gain.

**Writer level,** the measurement issue #49 asks for, reproduced against both
versions built from source on this machine (200 frames, `.mp4`/`Auto`,
gradient-with-blocks content). Figures are not comparable to #49's table, which
was measured on an i5-11400H:

| Resolution | v0.5.0 `writeFrame` mean / p95 | v0.6.0 `writeFrame` mean / p95 |
|---|---|---|
| 720p | 3.76 / 5.52 ms | 3.14 / 4.26 ms |
| 1080p | 8.64 / 11.27 ms | 7.24 / 9.60 ms |
| 1440p | 15.23 / 19.22 ms | 12.78 / 16.91 ms |

With nothing else on the caller's thread the encoder is the bottleneck either
way, so `writeFrame()` simply waits — which is the intended backpressure. The
gain appears once the caller has work to overlap, as the frame loop does
(150 frames, simulated per-frame inference on the calling thread):

| Case | v0.5.0 | v0.6.0 |
|---|---|---|
| 1080p, 20 ms/frame of work | 8.78 ms per write, **33.7 fps** | 0.03 ms per write, **47.0 fps** |
| 1440p, 25 ms/frame of work | 15.24 ms per write, **23.9 fps** | 0.01 ms per write, **37.2 fps** |

Encoding overlaps inference instead of serializing behind it: +39% and +56%
end-to-end throughput, and the per-call cost on the frame loop drops to the
hand-off. `release()` correspondingly grows (14–85 ms) because the queue drains
there, which is why finalization happens before the source is counted.

### M-4 — links

Every relative link added to `specs/roadmap.md` and this packet resolves
(`ls`-checked); `https://github.com/olibartfast/neuriplo-infer/issues/49`
returns `HTTP/2 200`.

### Deviations summary

1. No end-to-end finalize-failure fixture; covered at the seam through an
   injectable writer, with a negative control (V-4).
2. `pre-commit` unavailable; its tools run directly (V-8).
3. Generated-docs check run against the pinned README rather than the stale
   local `build/_deps` copy (V-9).
4. Early-exit path verified by unit test rather than an interactive `q` (M-2).
5. Builds run in a scratch worktree because the machine's `../neuriplo-tasks`
   sibling is stale — pre-existing, unrelated to this branch.
