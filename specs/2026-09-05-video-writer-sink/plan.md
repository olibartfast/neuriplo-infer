# Feature Plan — Video writer sink (`--output_video`)

> Derived from [`requirements.md`](requirements.md). Phases are in dependency
> order; each ends with the tree building and its phase-local check green. The
> default build stays unchanged until the writer is explicitly requested.

## Group 1 — Packet (orchestrator)

1. Create `specs/2026-09-05-video-writer-sink/` with the three artifacts,
   branch `feature/video-writer-sink` from `develop`. Nothing else happens in
   this phase.

## Group 2 — Foundation (pin and CMake)

2. **Pin `videocapture` to `v0.5.0`** in `versions.env`
   (`VIDEOCAPTURE_VERSION=v0.4.0` → `v0.5.0`).

3. **Add the opt-in switch** in `CMakeLists.txt`:
   - `option(NEURIPLO_INFER_WITH_VIDEOWRITER "Build the annotated-video writer
     sink (videocapture USE_VIDEOWRITER)" OFF)` near the other build-mode
     options.
   - Before `FetchContent_MakeAvailable(VideoCapture neuriplo-tasks
     nlohmann_json)`, the mapping:
     ```cmake
     if(NEURIPLO_INFER_WITH_VIDEOWRITER)
         set(USE_VIDEOWRITER ON CACHE BOOL "" FORCE)
     endif()
     ```
     This turns on videocapture's own `USE_VIDEOWRITER`, which builds the
     writer module and publishes the PUBLIC `VIDEOCAPTURE_WITH_WRITER` define.

Phase-local validation:

```bash
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release 2>&1 | tee /tmp/cfg-default.log
grep -q "VideoCapture writer: OFF" /tmp/cfg-default.log && cmake --build build
cmake -S . -B build-writer -DDEFAULT_BACKEND=OPENCV_DNN -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tee /tmp/cfg-writer.log
grep -q "VideoCapture writer: ON" /tmp/cfg-writer.log && cmake --build build-writer
```

## Group 3 — Behavior

4. **Add `videocapture::Frame toFrame(const cv::Mat&)`** to
   `FrameConversion.hpp` / `.cpp` — the inverse of `toBgrMat`. It accepts the
   packed BGR8 `CV_8UC3` layout the renderer guarantees, copies into a tightly
   packed `Frame(width, height, PixelFormat::BGR8)`, and rejects any other
   type with `std::invalid_argument` (mirroring `toBgrMat`'s rejection of
   unknown formats). An empty Mat converts to an empty Frame. `Frame` owns its
   storage, so this is always a copy (Deliberate, see requirements).

5. **`AppConfig`** (`app/inc/AppConfig.hpp`): add `std::string output_video;`
   beside the other run-artifact fields.

6. **Parser** (`app/src/CommandLineParser.cpp`): parse
   `{ output_video | | write annotated video to this file path }` into
   `config.output_video`, and validate in this order:
   - `output_video` set + image source → `exit(1)`:
     `--output_video applies to video sources only; image sources write stills`
     (unconditional, every build).
   - `output_video` set + `!defined(VIDEOCAPTURE_WITH_WRITER)` → `exit(1)`:
     `--output_video requires a build with
     -DNEURIPLO_INFER_WITH_VIDEOWRITER=ON`.

7. **Pipeline** (`app/src/CLICommands.cpp`): behind
   `#ifdef VIDEOCAPTURE_WITH_WRITER`, add a small RAII
   `VideoWriterInterface` sink (create + initialize + write + guaranteed
   `release()` in its destructor). In `processVideo` and
   `processVideoClassification`, when `config.output_video` is set:
   - initialize the sink from the first frame's width/height,
     `frameRate = 30.0`, `codec = VideoCodec::Auto`, destination
     `config.output_video`; on failure throw naming the flag, destination, and
     (odd-dimension H.264 case) the cause;
   - write the rendered frame — `toFrame(image)` in `processVideo`,
     `toFrame(displayFrame)` in `processVideoClassification` — inside the
     render stage scope, so writer time is attributed to `Render` like the
     still-path's PNG write;
   - `writeFrame()` returning false throws (no silent truncation).

8. **Capabilities** (`app/src/Capabilities.cpp`): under
   `#ifdef VIDEOCAPTURE_WITH_WRITER`, add `output_video`
   (`cli_flag "output_video"`, `value_type "path"`) to `parameterCatalog()`
   and to the `optional_parameters` of the task capabilities whose
   `source_types` include `"video"` and that render annotated video
   (object_detection, instance_segmentation, classification,
   video_classification, pose_estimation, depth_estimation,
   open_vocabulary_detection). Image-only tasks (optical_flow,
   gaussian_splatting) and image_understanding (which renders no video) are
   excluded. Both additions gate on the same macro, so
   `ParameterReferencesResolve` stays green in every build.

Phase-local validation:

```bash
cmake --build build        # default build still compiles with the flag parsed
cmake --build build-test-writer
./build-test-writer/<app-binary> --capabilities | grep -q '"output_video"'
```

## Group 4 — Contract and documentation

9. **Tests** (`app/test/`):
   - `test_FrameConversion.cpp`: `toFrame` round-trip (BGR8
     bytes/dims/format, non-`CV_8UC3` rejection, empty).
   - `test_parseCommandLineArguments.cpp`: default-empty, image-source
     rejection (both builds), and the writer-build-populates /
     writer-less-build-exits cases gated on `VIDEOCAPTURE_WITH_WRITER`.
   - Writer integration test (writer build only): write `.avi`, re-open via
     `createVideoInterface()`, assert dimensions.
   - Capabilities test: `output_video` advertised iff writer build (guarded
     assertion).

10. **Docs.** `README.md`: usage block line and an *Optional Parameters* bullet
    for `--output_video`, noting the `-DNEURIPLO_INFER_WITH_VIDEOWRITER=ON`
    requirement. No `Key Features` change (no new task type); no generated
    docs change (`sync_supported_model_types.py --check` must be clean).

11. **`CHANGELOG.md` [Unreleased]** — `Added`: `--output_video <path>` writes
    the annotated video to a file (30 fps, codec auto, container from
    extension), opt-in behind `-DNEURIPLO_INFER_WITH_VIDEOWRITER=ON`, image
    sources and writer-less builds rejected with an actionable error.
    `Changed`: pin `videocapture` `v0.4.0` → `v0.5.0`, which adds the writer
    sink module (no capture API change).

12. **`specs/roadmap.md`.** Remove the *Candidates* bullet for the
    video-writer sink; add a forward-dependency note to Phase 6
    (OpenCV-free application layer): when `FrameConversion` is deleted there,
    the writer bridge must be adapted (`cv::Mat → Frame` becomes
    `neuriplo_tasks::Image → Frame`).

## Group 5 — Verification

13. Execute [`validation.md`](validation.md) from the file, not from memory:
    the three-build scoreboard, the decisive CLI check, and the automated
    suite. Record deviations honestly in `validation.md`.

## Review and merge

14. Merge per
    [`../procedures/merge-feature-branch.md`](../procedures/merge-feature-branch.md):
    sync with `develop`, re-run the scoreboard after the merge, open the PR
    `feature/video-writer-sink` → `develop`, merge as a merge commit, delete
    the branch. No release tag on this branch; the changelog `[Unreleased]`
    entries ship with the next release.

> Commit by phase — Phase 1 (pin + CMake) and Phase 3 (CLI + pipeline) are the
> two commits a reviewer will want isolated; Phase 2 and 4 can fold with their
> neighbors.
