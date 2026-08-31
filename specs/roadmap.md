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
- Blocked on: a maintainer decision about the live preview window
  (`cv::imshow`), which has no in-family replacement — see Open Question 1.

---

## Candidates — not scheduled

Real, observed, small enough not to need a packet until someone picks one up:

- `versions.env` has accumulated a duplicated pin-comment block on every release;
  `scripts/cut_release.sh` appends instead of replacing it.
- FP16/BF16 over gRPC works only with raw tensor contents; the `KSERVE_BINARY=0`
  fallback silently cannot carry them.

## Replanning

Revisit this file at the end of every feature (stage 7). Delivered work changes
what should come next: merge phases that turned out to be one change, split a
phase that grew a dependency, and delete a phase the work made unnecessary.
