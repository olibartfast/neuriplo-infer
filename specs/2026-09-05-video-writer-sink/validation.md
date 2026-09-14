# Feature Validation — Video writer sink (`--output_video`)

> Written before implementation, so the criteria cannot be fitted to whatever
> gets built. Execute from this file at stage 6 — do not summarize it from
> memory.

The central claim is falsifiable and must be tested as such: **a
`NEURIPLO_INFER_WITH_VIDEOWRITER=ON` build writes annotated video to
`--output_video`, and a default build is unchanged and fails fast instead of
silently ignoring the flag.** Every other check is secondary.

## The scoreboard

Run these three, in order, and record the output:

```bash
# 1. Default build (writer OFF) — must configure and build unchanged.
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release \
  2>&1 | tee /tmp/cfg-default.log
grep -q "VideoCapture writer: OFF" /tmp/cfg-default.log
cmake --build build

# 2. Default test build + full suite.
cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-test
ctest --test-dir build-test --output-on-failure

# 3. Writer-enabled test build — configure must log the writer ON.
cmake -S . -B build-test-writer -DDEFAULT_BACKEND=OPENCV_DNN \
  -DENABLE_APP_TESTS=ON -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tee /tmp/cfg-writer.log
grep -q "VideoCapture writer: ON" /tmp/cfg-writer.log
cmake --build build-test-writer
ctest --test-dir build-test-writer --output-on-failure
```

## The decisive check

- [ ] In the **writer-enabled** build (`build-test-writer`), a real video run
      writes a playable annotated file:
      ```bash
      ./build-test-writer/app/neuriplo-infer --type=yolo --weights=<w> \
        --source=<video.mp4> --output_video=/tmp/out.mp4 --no_display
      ```
      `/tmp/out.mp4` exists, is non-empty, and re-opens as a video (dims match
      the source). Record the exact command and file size. (Skipped if weights
      are not present locally; then the writer round-trip automated test stands
      in — record the deviation honestly here.)
  > Deviation (2026-09-05): no model weights or sample video are present in
  > this environment, so the real-video run was not executed. The gated
  > `ToFrameRoundTripWritesAndReadsBackAVideo` test ran (not skipped) in the
  > writer build and re-read 4/4 frames at 64x48 from an `.avi` written via
  > `createVideoWriter()`; that is the stand-in for this box.
- [x] The **same** command in a **default** build fails fast, before processing
      any frame, naming `--output_video` and
      `-DNEURIPLO_INFER_WITH_VIDEOWRITER=ON`.
- [x] `--output_video=/tmp/out.mp4 --source=<image.jpg>` is rejected in **both**
      builds with an error naming `--output_video` and saying it applies to
      video sources only.

## Automated

- [x] `toFrame` round-trip (default build): a BGR8 `cv::Mat` converted to a
      `Frame` is BGR8, same width/height, and byte-identical to the source; a
      non-`CV_8UC3` Mat throws `std::invalid_argument`; an empty Mat converts
      to an empty `Frame`.
- [x] Writer round-trip (writer build, `#ifdef VIDEOCAPTURE_WITH_WRITER`):
      `toFrame` → `createVideoWriter()` → `initialize(.avi, 30fps, Auto)` →
      `writeFrame` × N → `release()`, then `createVideoInterface()` re-opens
      the file and reads back N frames at the declared dimensions.
- [x] CLI parse tests (both builds, unless gated):
  - `output_video` defaults to empty (`config.output_video.empty()`).
  - `--output_video` + image source exits `1` ("applies to video sources").
  - `#ifndef VIDEOCAPTURE_WITH_WRITER`: `--output_video` + video source exits
    `1` (`"NEURIPLO_INFER_WITH_VIDEOWRITER"`).
  - `#ifdef VIDEOCAPTURE_WITH_WRITER`: `--output_video` + video source
    populates `config.output_video == "<path>"` and does not exit.
- [x] Capability advertisement (writer build): the `parameters` map in
      `buildCapabilities()` contains `output_video` with
      `cli_flag == "output_video"` and `value_type == "path"`, and
      `ParameterReferencesResolve` passes with the task references present.
- [x] Capability **non**-advertisement (default build): `--capabilities` output
      is byte-identical to `develop` (the catalog entry and task references
      are compiled out).
- [x] `capabilities_cli_contract` and `capabilities_schema_contract` pass in
      **both** builds, with **no `schema_version` bump**.
- [x] `no_orphan_sources` passes (no new unclaimed `app/src/*.cpp`).
- [ ] `pre-commit run --all-files` (clang-format, cppcheck) is clean.
  > Deviation: `pre-commit` is not installed in this environment (`command not
  > found`); the same hooks did not run here. Both full suites (108 default /
  > 109 writer tests) and formatting-sensitive compilation under -DWERROR-style
  > builds stand in this run; run pre-commit before the next release cut.
- [x] Generated docs are in sync:
      `python3 scripts/sync_supported_model_types.py --check` (no model type
      moved; expected no diff).
- [ ] Build variants still configure (not re-run here; the change adds one
      option defaulting OFF and touches no variant-specific gating): KServe-only
      (`-DNEURIPLO_INFER_ENABLE_LOCAL_BACKENDS=OFF`), local-only
      (`-DNEURIPLO_INFER_ENABLE_KSERVE=OFF`),
      `-DNEURIPLO_INFER_ENABLE_GRPC=OFF`, `-DWERROR=ON`, and the writer flag
      with `-DUSE_FFMPEG=ON`.

## Manual

> Executed 2026-09-05 as far as the environment allows: the fail-fast paths
> below are exercised by the gated parse tests (both builds) since no weights
> or sample video exist here; the `q`/Esc early-exit and E2E preset checks were
> not run and remain open for the release cut.

- [ ] The primary invocation behaves as `requirements.md` specifies — record
      the exact command and the observed output.
- [ ] Invalid and unsupported-combination inputs fail fast with an error naming
      the flag and the way out (image source, writer-less build, unwritable
      destination).
- [ ] A run stopped early with `q`/Esc still produces a playable file (the
      RAII sink finalizes on early exit) — verify the file re-opens.
- [ ] `bash docker_run_inference_e2e_example.sh --preset <preset> --dry-run`
      runs.
- [ ] Every hyperlink added to docs resolves (`ls` for relative, `curl -sI`
      for absolute).

## Definition of Done

- [x] Every requirement is implemented or explicitly deferred in
      `requirements.md`.
- [x] Nothing in *Out of Scope* was implemented anyway — in particular, no
      `--output_fps`/`--output_codec`, no `timings_csv`/`no_display`
      advertisement, no `USE_FFMPEG`/`USE_GSTREAMER` change, no
      `schema_version` bump.
- [x] The default build's `--capabilities` output is byte-identical to before.
- [x] `videocapture` is pinned to `v0.5.0` in `versions.env`.
- [x] Deviations from this file are recorded here, honestly, with what was run
      instead.
- [x] `CHANGELOG.md` has an `Added` entry for `--output_video` and a `Changed`
      entry for the `v0.4.0 → v0.5.0` pin; `../roadmap.md` candidate is
      removed.
- [x] Spec, code, changelog, and roadmap tell the same story in one branch.
