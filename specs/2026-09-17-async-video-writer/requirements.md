# Feature Requirements — async video writer (videocapture v0.6.0)

Roadmap phase: [Phase 7 — Move `--output_video` encoding off the frame loop](../roadmap.md#phase-7--move---output_video-encoding-off-the-frame-loop--in-progress)
Branch: `feature/videocapture-0.6.0`

## Goal

`--output_video` currently encodes every annotated frame synchronously on the
frame loop's thread, which caps end-to-end throughput well below what the
inference backends reach (about 11 / 23 / 39 ms per frame at 720p / 1080p /
1440p `.mp4` on an i5-11400H — [#49](https://github.com/olibartfast/neuriplo-infer/issues/49)),
and a container that fails to finalize is logged by the writer but never
reported, so a truncated file leaves the run exiting `0`. `videocapture`
v0.6.0 moves encoding behind a bounded queue on the writer's own thread and
makes `VideoWriterInterface::release()` report whether the destination was
completed. This branch moves the pin onto that release and adapts the app to
both changes: encoding leaves the frame loop, and a destination that fails to
encode or finalize fails the run.

## In Scope

- `VIDEOCAPTURE_VERSION` pinned to `v0.6.0` in `versions.env`.
- `OutputVideoSink` (`app/src/CLICommands.cpp`) reports finalization failure:
  an explicit close on the normal path that throws when `release()` returns
  `false`, with the destructor still releasing on the exception and early-exit
  paths. Both writer call sites — `processVideo` and
  `processVideoClassification` — close through it.
- Frames are handed to the writer through the new
  `writeFrame(videocapture::Frame&&)` overload, so the encoder thread takes the
  pixels rather than copying them.
- The comments and documentation that describe the write as synchronous
  frame-loop work are corrected, and the `[0.10.0]` "Known limitations" entries
  this release resolves are recorded as resolved in `[Unreleased]`.
- Build modes covered: `-DNEURIPLO_INFER_WITH_VIDEOWRITER=ON` (the writer path)
  and the default writer-less build, which must keep compiling untouched.

## Out of Scope

- **A separate writer queue-wait figure in the run report** (the last bullet of
  [#49](https://github.com/olibartfast/neuriplo-infer/issues/49)'s proposal).
  Backpressure now shows up inside the render stage rather than as its own
  number. It touches `RunReport`, the capabilities schema, and their tests, and
  is worth its own reviewable change; recorded as a follow-up in
  [`../roadmap.md`](../roadmap.md), not implemented here.
- **Source frame rate.** `--output_video` still writes at a fixed 30 fps:
  v0.6.0's `VideoCaptureInterface` still reports no frame rate, so the
  `[0.10.0]` limitation stands.
- **Mid-stream read failure.** Still indistinguishable from end of stream
  through the capture interface; unchanged by v0.6.0.
- **A frame-drop backpressure policy.** Upstream never drops accepted frames
  and `writeFrame()` waits instead; the app does not add a policy on top.
- **macOS.** v0.6.0's packaged macOS builds are capture-only; this repo lists
  macOS as untested and gains no coverage here beyond a documented note.

## Decisions

- **The destructor is not the place to report a failed file.** `release()` now
  returns a result worth acting on, but a destructor that throws during stack
  unwinding terminates the process. So the normal path closes explicitly
  (`OutputVideoSink::finish()`) and throws there; the destructor releases
  whatever is still open and logs instead of throwing, which keeps the
  early-exit (`q`/Escape) and exception paths finalizing the file as they do
  today.
- **A failed finalize is a failed run, not a warning.** It is the same class of
  fault as `writeFrame()` returning `false`, which already throws: the operator
  asked for a playable artifact and did not get one. Exit code and the run
  report must not say otherwise.
- **`finish()` runs before the sample is counted,** in the same place
  `output_sink.reset()` does today — producing the artifact is part of
  processing the source, so a source whose file could not be finalized is not
  a completed sample.
- **No app-side writer thread.** The queue and thread belong in `videocapture`,
  which now has them; adding a second layer in the app would duplicate the
  buffering and the ordering guarantee. This is why #49's proposal is satisfied
  by a pin move plus error propagation rather than by new app machinery.

## Constraints and Context

- Constitution: [`../tech-stack.md`](../tech-stack.md) — video backend
  behavior, priority, and codec selection belong to `videocapture`; the app
  bridges frames and reports failures. No new dependency is added here.
- Existing patterns: `OutputVideoSink` is the RAII wrapper that already owns
  writer lifetime; `FrameTimingsCsv::finish()` is the precedent for an explicit
  close that can fail loudly while the destructor stays quiet.
- `neuriplo_infer::toFrame()` already returns a `Frame` that owns its pixels
  (it copies out of the `cv::Mat`), so handing it to another thread is safe and
  moving it is the only change needed.
- Cross-repo impact: requires `videocapture` `v0.6.0`, tagged upstream
  (`ebc7208`). No `neuriplo`, `neuriplo-tasks`, or `neuriplo-kserve-client`
  move.
- Contract surfaces: no CLI flag, `--capabilities` output, or schema change —
  `--output_video` keeps its name, its value type, and its rejection rules. The
  only visible contract move is that a run whose destination could not be
  finalized now exits non-zero.

## Open Questions

- None outstanding. Both questions raised before planning were answered by the
  maintainer: the `v0.6.0` tag is published upstream, so the pin moves in this
  branch; and the queue-wait metric is deferred rather than built here.
