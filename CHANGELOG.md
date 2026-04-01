# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.1] - 2026-04-01

### Fixed
- Dependency ref selection now follows the vision-inference release line: `master` uses dependency `master`, all other branches use dependency `develop`
- Reject invalid `VERSION` contents early during CMake configure
- Require `--tokenizer_vocab` and `--tokenizer_merges` explicitly for open-vocabulary detection
- Include TensorFlow runtime libraries in the `libtensorflow` Docker runtime image
- Fix Docker E2E script TensorRT image naming and RAFT multi-frame input handling

## [0.2.0] - 2026-03-31

### Added
- TensorRT Docker build job in CI workflow
- OWLv2 open-vocabulary detection support (via vision-core)
- Dependency branch ref validation in CMake (ensures neuriplo, videocapture, vision-core target the same ref)

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
- Migrated from per-backend detector classes to unified `TaskInterface`/`TaskFactory` (via vision-core)

[Unreleased]: https://github.com/olibartfast/vision-inference/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/olibartfast/vision-inference/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/olibartfast/vision-inference/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/vision-inference/releases/tag/v0.1.0
