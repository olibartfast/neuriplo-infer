# Feature Requirements — Video writer sink (`--output_video`)

Roadmap phase: [Candidates — not scheduled → *Consume the video-writer sink*](../roadmap.md)
Branch: `feature/video-writer-sink`

## Goal

A video run can write its annotated output to a file, so the rendered
detections/overlays become a playable video instead of a preview window no one
headless can see. `--output_video <path>` is added to the two video capture
loops (`processVideo`, `processVideoClassification`), consuming the writer sink
that `videocapture` ships in `v0.5.0`. The writer reuses the capture backend's
own codecs, so it introduces no new dependency: with `-DUSE_FFMPEG=ON` it costs
nothing new, and in the default OpenCV build it rides the OpenCV writer already
linked for capture.

## In Scope

- `--output_video <path>` (snake_case, parallel to `--timings_csv` /
  `--segmentation_output`) on the **video source paths only**: `processVideo`
  (detection, instance segmentation, classification, pose, depth, open-vocab
  detection) and `processVideoClassification`. Image sources are unchanged and
  keep writing per-frame stills.
- A `cv::Mat → videocapture::Frame` bridge, `neuriplo_infer::toFrame()`, in
  `FrameConversion` (the inverse of `toBgrMat()`).
- A writer sink in `CLICommands.cpp` that creates, initializes, feeds, and
  finalizes a `VideoWriterInterface` with fixed parameters: **30 fps**,
  **`VideoCodec::Auto`**, **container inferred from the destination extension**.
  `--output_fps` / `--output_codec` are deferred (see Out of Scope).
- The opt-in build switch `NEURIPLO_INFER_WITH_VIDEOWRITER` (default OFF)
  mapping onto videocapture's `USE_VIDEOWRITER`.
- `videocapture` pinned to `v0.5.0` in `versions.env`.
- `--capabilities` advertises `output_video` **only in a writer-enabled build**
  (see Decision 3).
- `README.md` usage + optional-parameter docs, `CHANGELOG.md`, `../roadmap.md`.

## Out of Scope

- **`--output_fps` / `--output_codec`.** Fixed at 30 fps, `VideoCodec::Auto`,
  container from extension. A follow-up adds the flags once there is demand for
  them; the writer config already supports both.
- **Advertising the pre-existing un-advertised run-artifact flags**
  (`--timings_csv`, `--no_display`, and the reserved `--no_gif`). They stay out
  of `--capabilities`; fixing that is a separate concern, not smuggled into this
  branch.
- **The image-source still path.** It already writes output and needs nothing;
  no `output_video` behavior is added to `processImage` / `processOpticalFlow`.
- **`USE_FFMPEG` / `USE_GSTREAMER` wiring.** No change. The writer follows
  whatever capture backend the build already selected; this branch does not turn
  either on.
- **`schema_version` bump.** Adding `output_video` to the parameter map is an
  additive change the published schema already permits (`parameters` allows
  arbitrary `$defs/parameter` entries), so `docs/capabilities.schema.json` and
  its version stay put.
- **Image understanding / VLM video output.** `processImageUnderstanding` does
  not route through the video capture loops and is not given a writer.
- **A zero-copy writer path.** `toFrame` copies; `Frame` owns its storage and
  cannot alias a `cv::Mat`. The copy is one frame buffer per frame — negligible
  next to inference — and avoids re-introducing the aliasing-vs-conversion
  branch that `toBgrMat` was built to hide.

## Decisions

1. **`--output_video` (snake_case), mirroring `--timings_csv` /
   `--segmentation_output`.** Maintainer-confirmed. No short flag, no
   `--output-video` alias.

2. **The writer is opt-in at build time, default OFF.** New CMake option
   `NEURIPLO_INFER_WITH_VIDEOWRITER`; when ON it force-sets
   `USE_VIDEOWRITER ON CACHE BOOL "" FORCE` before
   `FetchContent_MakeAvailable(VideoCapture …)`, so videocapture builds its
   writer module and publishes `VIDEOCAPTURE_WITH_WRITER` (PUBLIC) on the
   `VideoCapture` target. The default build's dependency surface is unchanged:
   it does not gain the writer module or any codec beyond what capture already
   links. This mirrors the `NEURIPLO_INFER_WITH_DISPLAY` (default OFF) pattern
   already planned for the preview.

3. **`output_video` is advertised in `--capabilities` only when the writer is
   compiled in.** The mission's *honest capability reporting* rule says "what
   it advertises is what the binary can actually route"; a writer-less build
   cannot route `--output_video`, so it must not advertise it. Both the
   `parameterCatalog()` entry and the per-task `optional_parameters` references
   are gated on `VIDEOCAPTURE_WITH_WRITER`, so `ParameterReferencesResolve`
   stays satisfied in every configuration and the default build's
   `--capabilities` output is byte-identical to today.

4. **A writer-less build parses `--output_video` but fails fast.** The flag
   surface is stable across build configs; asking for it without the writer
   exits with an error naming the flag and
   `-DNEURIPLO_INFER_WITH_VIDEOWRITER=ON`. This is the same "compile-time
   capability absent → fail fast, name the flag and how to rebuild" rule the
   planned `NEURIPLO_INFER_WITH_DISPLAY` preview flag follows; the repo forbids
   silent CLI breakage. `--output_video` on an **image source** is rejected
   unconditionally, in any build, because image sources write stills.

5. **The writer fails fast rather than degrading.** `initialize()` returning
   false aborts the run with an error naming `--output_video`, the destination,
   and — for the odd-width/height H.264 case — the cause. `writeFrame()`
   returning false mid-run also aborts; a run must not report success having
   produced a truncated or corrupt video. `release()` is best-effort and safe
   to call twice / on a never-initialized writer.

6. **`release()` runs on every exit path.** Only after `release()` returns is
   the destination guaranteed a complete, playable file; the interface's
   destructor does not finalize. The writer is wrapped in a small RAII sink
   whose destructor calls `release()`, so a `q`/Esc early exit or an exception
   still finalizes the file.

## Constraints and Context

- **Repo boundary respected.** `videocapture` owns video-backend selection and
  codec behavior (`specs/tech-stack.md`: no video-backend selection policy
  here); this branch only passes a destination and a config through, exactly as
  it already passes source flags. The writer module lives in videocapture, not
  here.
- **Cross-repo sequencing.** `videocapture v0.5.0` must exist and be pinned
  (`versions.env`, a concrete `vX.Y.Z` tag — `scripts/validate_release_pins.sh`
  blocks a branch pin) before this branch can be released. Capture semantics
  are unchanged from `v0.4.0` (the `Frame` API is the same), so the pin is
  expected to be a drop-in; Phase 1 verifies that before any app change.
- **Frame/writer contract.** `writeFrame()` requires a packed 8-bit layout
  (`Gray8/RGB8/BGR8/RGBA8/BGRA8`) and the dimensions declared in the config.
  The renderer always produces packed BGR8 (`toBgrMat` guarantees it, and
  `displayFrame` is a clone of it), so `toFrame` only needs the BGR8 case.
  Dimensions are fixed for a video source, so the writer is configured once
  from the first frame.
- **Writer backend priority and the odd-dimension H.264 failure.** FFmpeg >
  GStreamer > OpenCV, same as capture. The FFmpeg writer rejects odd
  width/height for H.264 and does not auto-pad; that rejection surfaces as an
  `initialize()`/`writeFrame()` failure and becomes the fail-fast error from
  Decision 5, not a silent no-op.
- **Contract surfaces touched.** `AppConfig` gains one string field; the CLI
  params string gains one entry; `--capabilities` gains one parameter entry
  (writer builds only). No `schema_version` change;
  `capabilities_schema_contract` and `capabilities_cli_contract` must still
  pass, byte-identical in the default build.
- **`no_orphan_sources`.** No new `app/src/*.cpp` file is introduced —
  `toFrame` lands in the already-listed `FrameConversion.cpp`, the sink in the
  already-listed `CLICommands.cpp` — so the guard needs no change.
- **Forward dependency: `feature/opencv-free-app`.** That branch migrates
  rendering off `cv::Mat` and deletes `FrameConversion`. The writer path added
  here must then adapt: `cv::Mat → Frame` becomes
  `neuriplo_tasks::Image → Frame` (write the Image's packed BGR8 buffer into a
  `Frame`). This is recorded here and in the roadmap so the later branch does
  not delete `toFrame` and strand the writer.

## Open Questions

1. **Test codec/container.** The writer round-trip test writes to an `.avi`
   destination (`VideoCodec::Auto` → MJPEG), which the OpenCV writer supports
   without external encoders, and re-reads it with `createVideoInterface()`.
   If a CI runner's OpenCV `videoio` is built without even MJPEG/AVI, the test
   should skip itself (GTEST_SKIP) rather than fail — resolve at
   implementation time, not by weakening the assertion on a capable machine.
