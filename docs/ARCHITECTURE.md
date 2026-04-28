# Project Architecture

This repo is the application layer in the `vision-stack` cluster. It owns CLI parsing,
app configuration, runtime wiring, visualization, and end-to-end execution flow.

## Canonical Sources

- [`ops/CLUSTER_MAP.yaml`](../ops/CLUSTER_MAP.yaml): repo roles, dependency edges, validation order
- [`ops/repo-meta/vision-inference.yaml`](../ops/repo-meta/vision-inference.yaml): repo-local entrypoints, public surface, constraints
- [`CMakeLists.txt`](../CMakeLists.txt): actual build requirements, backend options, and fetched dependencies
- [`cmake/versions.cmake`](../cmake/versions.cmake): dependency-ref derivation and version-loading behavior
- [`docs/generated/supported-model-types.md`](generated/supported-model-types.md): generated upstream TaskFactory model-type inventory

## Repo Boundaries

- `vision-inference`: CLI, configuration, runtime wiring, visualization, end-to-end app flow
- `vision-core`: task contracts, preprocessing, postprocessing, result types, model-specific task logic
- `neuriplo`: backend abstractions, backend adapters, runtime compatibility, backend dependency versions
- `videocapture`: source semantics, file/stream/camera handling, video backend priority and behavior

Treat [`ops/CLUSTER_MAP.yaml`](../ops/CLUSTER_MAP.yaml) as the source of truth for these boundaries.

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
