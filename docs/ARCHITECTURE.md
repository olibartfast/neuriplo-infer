# Project Architecture

This repo is the application layer in the `vision-stack` cluster. It owns CLI parsing,
app configuration, runtime wiring, visualization, and end-to-end execution flow.

## Canonical Sources

- `vision-platform/ops/CLUSTER_MAP.yaml`: cross-repo roles, dependency edges, validation order
- `vision-platform/ops/repo-meta/vision-inference.yaml`: repo-local entrypoints, public surface, constraints
- [`CMakeLists.txt`](../CMakeLists.txt): actual build requirements, backend options, and fetched dependencies
- [`cmake/versions.cmake`](../cmake/versions.cmake): dependency-ref derivation and version-loading behavior
- [`docs/generated/supported-model-types.md`](generated/supported-model-types.md): generated upstream TaskFactory model-type inventory

## Repo Boundaries

- `vision-inference`: CLI, configuration, runtime wiring, visualization, end-to-end app flow
- `vision-core`: task contracts, preprocessing, postprocessing, result types, model-specific task logic
- `neuriplo`: backend abstractions, backend adapters, runtime compatibility, backend dependency versions
- `videocapture`: source semantics, file/stream/camera handling, video backend priority and behavior

Treat `vision-platform/ops/CLUSTER_MAP.yaml` as the source of truth for cross-repo boundaries.

## What This Repo Intentionally Does Not Own

- Tensor shapes, dtype semantics, and result-schema meaning
- Backend implementation details or backend package versions
- Video backend selection policy beyond wiring through `videocapture`
- Upstream model-type inventory beyond the subset this app routes and validates end to end

Those contracts live in sibling repos and should not be redefined here in hand-maintained prose.

## Build and Dependency Notes

- The build currently requires CMake 3.24 and C++20. Read these from [`CMakeLists.txt`](../CMakeLists.txt), not from copied snippets in docs.
- This repo derives a shared dependency ref for `vision-core`, `neuriplo`, and `videocapture` in [`cmake/versions.cmake`](../cmake/versions.cmake).
- Backend package versions such as ONNX Runtime, TensorRT, LibTorch, OpenVINO, TensorFlow, and CUDA are owned by `neuriplo` and consumed through setup/build wiring here.

## Model Types

The generated list in [`docs/generated/supported-model-types.md`](generated/supported-model-types.md)
reflects the upstream `vision-core` TaskFactory inventory.

That list is broader than the guarantees made by this application repo. End-to-end behavior
still depends on the app's own CLI validation, task routing, rendering, and test coverage.

## Runtime Flow

The app runtime is split by responsibility:

- `CommandLineParser`: parses CLI flags and validates app-owned input paths and task-specific assets.
- `VisionApp`: sets up logging, builds an `InferencePipeline`, and selects a CLI command.
- `InferencePipelineBuilder`: wires labels, backend engine, model metadata, `vision-core` task config, task instance, and result renderer.
- `CLICommands`: owns executable workflows such as normal inference, warmup, benchmarking, image-understanding dispatch, and metadata export.
- `ResultRenderer`: renders `vision-core::Result` variants for each routed `TaskType`.
- `TaskRouting`: maps app model-type strings to `vision_core::TaskType` and must stay aligned with `vision_core::TaskFactory`.

`--export_metadata` follows the same backend and task setup path as inference, then prints model type, routed task type, input layers, and output layers before exiting. It intentionally does not require a source.

## Local Quality Checks

Repo-local validation remains the canonical CMake test flow from ops metadata. For deeper local checks, [`scripts/check_code_quality.sh`](../scripts/check_code_quality.sh) runs format checking, `cppcheck`, an AddressSanitizer/UndefinedBehaviorSanitizer test build, and a ThreadSanitizer test build when those tools are installed.
