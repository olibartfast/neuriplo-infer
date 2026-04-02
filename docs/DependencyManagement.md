# Dependency Management Guide

This document describes which files are authoritative for dependency behavior in
`vision-inference` and which files are only convenience tooling.

## Source of Truth

- [`CMakeLists.txt`](../CMakeLists.txt): build requirements, backend options, FetchContent wiring
- [`cmake/versions.cmake`](../cmake/versions.cmake): version loading and shared dependency-ref selection
- [`versions.env`](../versions.env): repo-owned minimum system dependency versions
- [`ops/repo-meta/vision-inference.yaml`](../ops/repo-meta/vision-inference.yaml): canonical configure/build/test commands
- `neuriplo`: backend package versions for ONNX Runtime, TensorRT, LibTorch, OpenVINO, TensorFlow, CUDA
- `videocapture`: video-backend dependency details and backend-specific setup guidance

## What This Repo Owns

- Minimum system requirements used by this app build: OpenCV, glog, and CMake
- Which sibling repos are fetched and how their ref is selected
- The top-level setup scripts that help users install backend dependencies locally

## What This Repo Does Not Own

- Backend package version numbers
- Backend linking details
- Video-backend dependency matrices
- Upstream task/model contracts

Those belong to `neuriplo`, `videocapture`, and `vision-core`.

## Shared Dependency Ref

`vision-inference` does not manually pin independent versions for `vision-core`,
`neuriplo`, and `videocapture` in prose docs.

Instead, [`cmake/versions.cmake`](../cmake/versions.cmake) derives one shared ref:

- `master` release builds resolve sibling repos to `master`
- all other branches resolve sibling repos to `develop`

If explicit per-repo overrides disagree with that derived ref, configure fails.

## Setup Scripts

The scripts under [`scripts/`](../scripts/) are convenience tooling, not the canonical
definition of dependency policy.

- [`scripts/setup_dependencies.sh`](../scripts/setup_dependencies.sh): installs selected backend dependencies for local development/runtime setup
- [`scripts/update_backend_versions.sh`](../scripts/update_backend_versions.sh): helper that copies backend-version files from sibling/fetched repos when needed by setup scripts

These scripts may read files such as `versions.neuriplo.env` for backend setup, but those
files are not the source of truth for which sibling repo refs the main CMake build targets.

## Recommended Workflow

1. Use the canonical commands in [`ops/repo-meta/vision-inference.yaml`](../ops/repo-meta/vision-inference.yaml) for configure/build/test.
2. Run a setup script only when you need a backend dependency installed locally.
3. When reasoning about repo compatibility, prefer [`cmake/versions.cmake`](../cmake/versions.cmake) and [`ops/CLUSTER_MAP.yaml`](../ops/CLUSTER_MAP.yaml) over copied version tables.
4. For video-backend setup specifics, consult `videocapture` documentation instead of duplicating those instructions here.

## Platform Notes

- Linux is the primary supported environment for this repo and its helper scripts.
- Other platforms may work for subsets of the stack, but their exact backend support should be treated as repo-specific and verified in the owning dependency repo.
