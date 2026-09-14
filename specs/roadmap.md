# Roadmap

The smallest sensible delivery order for work that is not yet done. Each phase is
independently reviewable and testable, and gets its own dated feature packet
under `specs/YYYY-MM-DD-<feature-name>/` when it starts.

This is a brownfield roadmap: it starts from unfinished work, not from phase one
of the product. Shipped behavior is recorded in [`CHANGELOG.md`](../CHANGELOG.md),
not here.

Status: `not started` · `in progress` · `blocked` · `done`

---

## Phase 1 — Spec-driven workflow adopted · in progress

Constitution files, feature-packet templates, and the procedure docs live in
`specs/`, and `AGENTS.md` points an agent at them before planning.

- Packet: this branch (`feature/spec-driven-workflow`)
- Done when: `mission.md`, `tech-stack.md`, `roadmap.md`, and `templates/` exist;
  `AGENTS.md` and `README.md` link them; the first real feature packet is written
  from the templates rather than from prose.

## Phase 2 — Cut the release carrying the ensemble work · not started

`CHANGELOG.md` `[Unreleased]` holds the server-side ensemble mode, the YOLO26
depth routing, `--segmentation_output`, and the neuriplo-tasks v0.8.0 pin. That is
a release's worth of user-visible change sitting on `develop`.

- Depends on: nothing in this list
- Done when: a `release/*` branch bumps `VERSION` and pins via
  `scripts/cut_release.sh`, `scripts/validate_release_pins.sh` passes, the tag is
  pushed, Publish GitHub Release has run, and
  `git rev-list --left-right --count origin/develop...origin/master` is `0 0`.
- After the release is published: close issues
  [#38](https://github.com/olibartfast/neuriplo-infer/issues/38) and
  [#39](https://github.com/olibartfast/neuriplo-infer/issues/39) if the README
  quickstart / docs split is in it, replying in English and Chinese with links to
  the README quickstart, platform table, and `docs/Deployment.md`. Not on merge to
  `develop` — the maintainer wants them closed only by a release.

## Phase 3 — Typed errors in `neuriplo-kserve-client` · not started

Both clients throw a plain `std::runtime_error` with the HTTP status or gRPC code
flattened into the message (`KserveHttpClient.cpp:513`, `throwStatus` at
`KserveGrpcClient.cpp:75`), so a consumer cannot tell "the server does not support
this operation" from "that model does not exist" without matching on message text.
Phase 4 needs that distinction; string-matching another repo's error prose is not
an acceptable substitute.

- Owned by [neuriplo-kserve-client](https://github.com/olibartfast/neuriplo-kserve-client),
  not this repo. Raise it there; this phase tracks the dependency.
- Contract this repo needs: a client exception type that carries the transport
  status and a transport-agnostic classification, so a caller can distinguish an
  unsupported operation (HTTP 4xx / gRPC `UNIMPLEMENTED`) from other failures
  without parsing text. The exact shape is that repo's design call.
- Done when: the client ships the typed error in a tagged release,
  `NEURIPLO_KSERVE_CLIENT_VERSION` in `versions.env` moves to it, and the build
  and existing KServe tests pass on the new pin.

## Phase 4 — Expose KServe model management through the CLI · not started

Depends on Phase 3.

Model Repository index / load / unload is implemented on the client API for both
transports but is unreachable from the command line (see *Known Limitations* in
`README.md`).

- Packet: [`2026-08-29-kserve-repository-cli/`](2026-08-29-kserve-repository-cli/)
  — specified and interviewed; blocked on Phase 3. Shape settled on an action flag
  `--kserve_repository=index|load|unload`, all three operations, and classifying
  an extension-disabled failure rather than probing for it.
- Done when: the operations are reachable, an unsupported-operation failure is
  named as such via the typed error from Phase 3, covered by tests, and reflected
  in `--capabilities` and `docs/KserveRuntime.md`.

## Phase 5 — Make Triton/OVMS compatibility evidence routine · not started

Triton and OVMS round-trips currently run as a CI dry-run per PR, with live runs
behind manual dispatch. The compatibility matrix in `docs/KserveCompatibility.md`
is therefore only as fresh as the last manual run.

- Done when: live runs happen on a declared cadence (schedule or release gate),
  and the matrix records the run that produced each cell.

---

## Phase 6 — OpenCV-free application layer · not started

`neuriplo` confines OpenCV to its `OPENCV_DNN` backend, `neuriplo-tasks` makes it
an optional adapter (default off), and `videocapture` v0.4.0 took `cv::Mat` out
of its frame API. This repo is the last holdout: `find_package(OpenCV REQUIRED)`
is unconditional, so a TensorRT or KServe-only build still needs OpenCV that no
sibling asked for. The blocker is rendering — ~70 of the ~140 `cv::` uses in
`app/` draw overlays, and nothing in the family draws.

Packet: [`2026-08-31-opencv-free-app/`](2026-08-31-opencv-free-app/requirements.md).

- Done when: a non-`OPENCV_DNN` image builds and runs with no OpenCV package
  installed, `ldd` on its binary lists no `libopencv_*`, and the rendered output
  is unchanged for every task type.
- The live preview survives as SDL2 behind `NEURIPLO_INFER_WITH_DISPLAY`,
  default OFF — an approved, bounded exception to the no-new-dependency rule,
  so no headless image pays for it.
- The video writer bridge added by the `--output_video` feature
  (`neuriplo_infer::toFrame`, `cv::Mat → Frame`) must be adapted to
  `neuriplo_tasks::Image → Frame` in the same step that deletes
  `FrameConversion`, or the writer path stops compiling on the way to
  OpenCV-free.

---

## Candidates — not scheduled

Real, observed, small enough not to need a packet until someone picks one up:

- `versions.env` has accumulated a duplicated pin-comment block on every release;
  `scripts/cut_release.sh` appends instead of replacing it.
- FP16/BF16 over gRPC works only with raw tensor contents; the `KSERVE_BINARY=0`
  fallback silently cannot carry them.
- The Docker images unpack backend SDKs under `/root`, so
  `docker run --user "$(id -u):$(id -g)"` fails to load them
  (`libonnxruntime.so.1: cannot open shared object file`) and host output is
  root-owned. Found while verifying the README quickstart (2026-09-11).
- `OPENCV_DNN` built against Ubuntu 24.04's apt OpenCV 4.6 rejects current
  Ultralytics ONNX exports (YOLO11n attention `Split`; YOLOv8n `Unsqueeze`, also
  at opset 12), so the default backend cannot run the most common first model.
- `versions.env` `CMAKE_MIN_VERSION=3.20` disagrees with
  `cmake_minimum_required(VERSION 3.24)`.

Done outside a phase: README quickstart and user-docs split
([`2026-09-11-readme-quickstart-docs/`](2026-09-11-readme-quickstart-docs/requirements.md)),
addressing issues #38 and #39, which close with the next release (Phase 2).

## Replanning

Revisit this file at the end of every feature (stage 7). Delivered work changes
what should come next: merge phases that turned out to be one change, split a
phase that grew a dependency, and delete a phase the work made unnecessary.
