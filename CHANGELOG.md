# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.6.2] - 2026-06-13

### Added
- CLI output images are named `processed_<model>_<backend>.png` instead of a
  fixed `processed.png`, making multi-run local output easier to distinguish.

### Changed
- Sibling release pin bumped: `neuriplo` v0.6.0 -> v0.7.0 (tensor datatype
  metadata on `InferenceMetadata`). `videocapture` (v0.3.0), `neuriplo-tasks`
  (v0.4.0), and `neuriplo-kserve-client` (v0.3.0) unchanged.
- Compatibility matrix: `neuriplo-kserve-runtime` gRPC transport validated
  against the v0.2.0 runtime release (binary tensor framing and real tensor
  datatypes in model metadata).

### Removed
- Auto-publish GitHub Release workflow (`.github/workflows/publish-github-release.yml`);
  release notes are published manually after Release Guard validates sibling pins.

## [0.6.1] - 2026-06-12

### Fixed
- `InferencePipelineBuilderTest` adapted to the neuriplo v0.6.0 load-failure
  contract: `setup_inference_engine` no longer lets vendor exceptions
  (e.g. `cv::Exception`) propagate -- it logs them and returns `nullptr`, so
  the builder's own `runtime_error` is now the expected failure shape for the
  intentionally-unparseable YOLO26 model under OpenCV 4.6. Fixes the red
  `master Release Check` / `develop CI` test jobs after the v0.6.0 release.

## [0.6.0] - 2026-06-12

### Changed
- Sibling release pins bumped: `neuriplo` v0.5.0 -> v0.6.0 (multi-backend
  builds, dlopen plugin ABI, raw typed-buffer output API) and
  `neuriplo-kserve-client` v0.1.0 -> v0.3.0 (proto profiles, gRPC
  raw-contents conformance). `videocapture` (v0.3.0) and `neuriplo-tasks`
  (v0.4.0) unchanged.
- Compatibility matrix: `neuriplo-kserve-runtime` gRPC transport is
  live-validated against the v0.1.0 runtime release
  (`raw_output_contents` emitted by default).

## [0.5.0] - 2026-06-11

### Added
- KServe V2 remote runtime mode: run task preprocessing/postprocessing locally
  while sending inference tensors to a KServe V2 endpoint (`--kserve_endpoint`,
  `--kserve_model_name`, `--kserve_transport`, etc.), with HTTP and optional gRPC
  transport, TLS/mTLS, bearer auth, retry/backoff, keep-alive, binary tensor
  extension, readiness probing, and integration tests. See `docs/KserveRuntime.md`
  and `docs/KserveCompatibility.md`.
- KServe V2 Model Repository extension on the runtime client (now in
  `neuriplo-kserve-client`): `IClient` gains `repositoryIndex()` /
  `loadModel(name)` / `unloadModel(name)` as an optional capability (base methods
  throw; HTTP and gRPC clients implement them). HTTP POSTs
  `/v2/repository/index|models/{m}/load|models/{m}/unload`; gRPC adds the
  `RepositoryIndex` / `RepositoryModelLoad` / `RepositoryModelUnload` RPCs
  (field numbers matching the official KServe/Triton service). Pure path
  builders, the neutral `RepositoryModel` result, and `parseRepositoryIndex`
  are unit-tested; the calls reuse the existing retry/auth/TLS plumbing. See
  `docs/KserveRuntime.md`.

### Changed
- Closed out the KServe production roadmap: `feature/neuriplo-kserve-runtime`
  merged into `develop` (2026-06-09) with all phases complete. The roadmap doc
  was then transformed into `docs/KserveRuntime.md`, a reference for agents and
  humans (architecture, capabilities, configuration/env vars, build modes,
  testing); the historical gap tables and phase checklists were removed
  (CHANGELOG and git history keep the record). README and
  `docs/KserveCompatibility.md` updated to match actual capability (FP16/BF16
  over gRPC via the default raw tensor contents) and now document the
  `KSERVE_BINARY` and `KSERVE_MAX_RETRIES`/`KSERVE_RETRY_*` environment
  variables.
- Extracted the KServe V2 protocol client into a standalone sibling repository,
  [`neuriplo-kserve-client`](https://github.com/olibartfast/neuriplo-kserve-client)
  (the pure, backend-agnostic HTTP/gRPC client + proto + protocol/retry/security
  unit tests). neuriplo-infer now consumes it via `FetchContent`, pinned by
  `NEURIPLO_KSERVE_CLIENT_VERSION` in `versions.env` (`v0.1.0`) and governed by
  the same release tooling as the other siblings. Only the `KserveEngine`
  adapter (which bridges the client to the neuriplo inference contract) and its
  test remain in this repo. The library's gRPC-availability signal is the PUBLIC
  `KSERVE_CLIENT_WITH_GRPC` define (replaces the in-tree `NEURIPLO_INFER_WITH_GRPC`
  / `NEURIPLO_INFER_WITH_KSERVE_TLS` build flags).
- Removed the deprecated Acknowledgments section from `README.md`.

### Fixed
- CLI now exits cleanly when `--source` is missing instead of surfacing a
  confusing downstream error.

## [0.4.1] - 2026-06-07

### Changed
- Slimmed `README.md` and removed deprecated docs (`PROGRESS.md`,
  `docs/Roadmap.md`, `docs/FasterCompilationPlan.md`) and the deprecated local
  `ops/` compatibility pointer; redirected agent references to
  `neuriplo-platform/ops`.
- Dropped the stale `neuriplo-tasks` branch pin in the OPENCV_DNN CI job.

### Fixed
- Removed the stale YOLO26 LiteRT Docker build override that pinned `neuriplo`
  to commit `8cf93e6`, so release builds use the `versions.env` pin
  `NEURIPLO_VERSION=v0.5.0` with the LiteRT NCHW→NHWC transpose fix.
- Moved LiteRT dependency setup before the full source copy in the Docker build
  so source-only changes do not force a TensorFlow Lite rebuild.

## [0.4.0] - 2026-06-07

### Changed
- Renamed application entry class to `NeuriploInfer` and aligned related
  source/test filenames with the `NeuriploInfer*` prefix.
- Renamed repository identity per ADR 0004: `neuriplo-infer` CMake project,
  static library, and executable output name, sibling pin
  `NEURIPLO_TASKS_VERSION`, and consumer updates for `neuriplo-tasks`
  includes/namespaces/link targets. The GitHub repo rename
  (`vision-inference` → `neuriplo-infer`) and the `vision-core` →
  `neuriplo-tasks` rename are both complete; FetchContent and the release
  scripts now track the renamed repos directly and the legacy compat shims
  were removed.
- Pinned `neuriplo` to `v0.5.0` (Abstract-Factory backend features),
  `neuriplo-tasks` to `v0.4.0`, and `videocapture` to `v0.3.0`.

## [0.3.2] - 2026-05-28

### Changed
- Pinned `neuriplo` to `v0.4.0` (LiteRT NCHW→NHWC transpose fix, new LiteRT
  backend) and `neuriplo-tasks` to `v0.3.2` (YOLO26 normalized coordinate scaling
  fix). `videocapture` pin unchanged at `v0.2.0`.

## [0.3.1] - 2026-05-21

### Changed
- Pinned `neuriplo-tasks` to `v0.3.1` in `versions.env` (README ↔ TaskFactory contract fixes and test hardening; no API change). `neuriplo` and `videocapture` pins unchanged.
- Re-synced `README.md` and `docs/generated/supported-model-types.md` from neuriplo-tasks v0.3.1's model-type block

## [0.3.0] - 2026-05-21

### Added
- Gemma4 VLM image understanding wired via llama.cpp + libmtmd
- Grounding DINO open-vocabulary detection wired as `OpenVocabDetection` with BERT tokenizer support
- Multimodal CLI parameters and task routing
- VLM image understanding how-to and agentic wiring runbook

### Changed
- Sibling refs (`neuriplo`, `neuriplo-tasks`, `videocapture`) pinned in `versions.env` to each sibling's own release tag for reproducible tag checkouts; siblings now version independently
- Release tooling reworked for per-sibling pins: `cut_release.sh` detects each sibling's latest release tag, `validate_release_pins.sh` rejects branch-name pins (e.g. `master`/`develop`), enforced by the pre-push hook and the release-guard workflow
- Supported model types synced from neuriplo-tasks (adds ImageUnderstanding)
- CI skips on docs-only pushes; frees disk space before Docker builds

### Fixed
- Pre-push hook no longer runs `act` (no locally-runnable CI jobs for this repo)

## [0.2.3] - 2026-04-03

### Fixed
- Restored build dependency fetching through declared refs instead of relying on local sibling checkouts

## [0.2.2] - 2026-04-03

### Changed
- Derive the shared dependency ref inside CMake and release-aware tooling instead of carrying `DEPENDENCIES_VERSION` in `versions.env`
- Single source of truth for model types via sync script (`sync_supported_model_types.py`)
- Sync generated supported model types into `README.md` and `docs/generated/supported-model-types.md`
- Added `--check` dry-run mode to sync script for CI drift detection
- Added CI step to verify model-type docs are in sync after cmake configure
- Consolidated agent guidance into `AGENTS.md` and replaced helper instruction files with links to the canonical source
- Reduced duplicated documentation by rewriting architecture and dependency docs around canonical sources of truth

### Fixed
- Removed the stale hand-maintained compatibility matrix and replaced its remaining references with generated or code-owned sources

## [0.2.1] - 2026-04-01

### Fixed
- Dependency ref selection now follows the neuriplo-infer release line: `master` uses dependency `master`, all other branches use dependency `develop`
- Reject invalid `VERSION` contents early during CMake configure
- Require `--tokenizer_vocab` and `--tokenizer_merges` explicitly for open-vocabulary detection
- Include TensorFlow runtime libraries in the `libtensorflow` Docker runtime image
- Fix Docker E2E script TensorRT image naming and RAFT multi-frame input handling
- Fix CI backend version fetches for branch-specific dependency refs
- Fix Docker CI builds for LibTensorFlow, TensorRT, and OpenVINO on the release branch
- Honor explicit `DEPENDENCIES_VERSION` overrides in non-git build contexts such as Docker

## [0.2.0] - 2026-03-31

### Added
- TensorRT Docker build job in CI workflow
- OWLv2 open-vocabulary detection support (via neuriplo-tasks)
- Dependency branch ref validation in CMake (ensures neuriplo, videocapture, neuriplo-tasks target the same ref)

### Fixed
- CMake validation function name collision with neuriplo (renamed to `validate_project_dependencies`)
- Dockerfile.libtensorflow pip-based build and trailing newline issues
- Dependency ref resolution and GitHub Actions branch detection

### Changed
- Unified dependency versioning via `DEPENDENCIES_VERSION` in `versions.env` (replaces per-library version pins)
- CI workflow targets `develop` branch instead of `main`/`master`

## [0.1.0] - 2026-03-02

### Added
- Confidence, NMS, and mask threshold CLI flags (`--confidence`, `--nms`, `--mask_threshold`)
- Threshold passthrough from CLI → `AppConfig` → `TaskConfig` → task constructors
- Depth estimation task support
- TensorRT precision option in inference scripts
- Docker end-to-end example scripts
- Composite GitHub Action to fetch neuriplo `versions.env`
- OpenVINO and LibTensorflow Docker CI builds
- GTest-based unit test suite (CLI parsing, threshold mapping, utils)
- Docker builds no longer depend on pre-existing `build/_deps/neuriplo-src/versions.env` (#15)
- Confidence/NMS/mask thresholds now correctly passed from CLI to task factory (#18)
- Dockerfiles source backend versions from neuriplo `versions.env`
- Migrated from per-backend detector classes to unified `TaskInterface`/`TaskFactory` (via neuriplo-tasks)

[Unreleased]: https://github.com/olibartfast/neuriplo-infer/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/olibartfast/neuriplo-infer/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/olibartfast/neuriplo-infer/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/olibartfast/neuriplo-infer/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/olibartfast/neuriplo-infer/compare/v0.4.1...v0.5.0
[0.4.0]: https://github.com/olibartfast/neuriplo-infer/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/olibartfast/neuriplo-infer/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/olibartfast/neuriplo-infer/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/olibartfast/neuriplo-infer/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/olibartfast/neuriplo-infer/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/olibartfast/neuriplo-infer/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/olibartfast/neuriplo-infer/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/olibartfast/neuriplo-infer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/neuriplo-infer/releases/tag/v0.1.0
