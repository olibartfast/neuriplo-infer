# Dependency Management Guide

This document describes dependency behavior owned by `neuriplo-infer`: local
build inputs, app-level sibling checkout selection, and setup scripts.

## Source Of Truth

Repo-local sources:

- [`CMakeLists.txt`](../CMakeLists.txt): build requirements, backend options,
  FetchContent wiring.
- [`cmake/versions.cmake`](../cmake/versions.cmake): version loading and
  app-level sibling-ref selection.
- [`versions.env`](../versions.env): repo-owned minimum system dependency
  versions and concrete sibling release pins.

External owners:

- `neuriplo-tasks`: task contracts and result semantics.
- `neuriplo`: backend package versions and backend runtime compatibility.
- `videocapture`: video backend setup and source semantics.

## What This Repo Owns

`neuriplo-infer` owns:

- Minimum system requirements used by this app build, such as OpenCV, glog, and
  CMake.
- How the local app fetches and wires sibling repositories at build time.
- Top-level setup scripts that help users install backend dependencies locally.
- CLI-facing configuration for selecting tasks, sources, weights, labels,
  prompts, output format, batching, warmup, benchmarking, and metadata export.

## What This Repo Does Not Own

`neuriplo-infer` does not own:

- Backend package version numbers.
- Backend linking internals.
- Video-backend dependency matrices.
- Upstream task/model contracts.

Those belong to `neuriplo`, `videocapture`, and `neuriplo-tasks`.

## Sibling Ref Selection

[`cmake/versions.cmake`](../cmake/versions.cmake) resolves each sibling
independently, including `neuriplo-kserve-client`, in this order:

1. An explicit CMake `-D<REPO>_VERSION=<ref>` override.
2. The corresponding pin in [`versions.env`](../versions.env).
3. A shared fallback: `master` on the `master` branch, `develop` on other
   branches. Detached checkouts use `master` for a plain release version and
   `develop` otherwise.

Different sibling refs are supported and logged, not rejected. With concrete
pins present, switching the app branch does not replace them with branch refs.
Release pins must be concrete `vX.Y.Z` tags; see [Versioning.md](Versioning.md).

## Setup Scripts

The scripts under [`scripts/`](../scripts/) are convenience tooling, not the
canonical definition of dependency policy.

- [`scripts/setup_dependencies.sh`](../scripts/setup_dependencies.sh): installs
  selected backend dependencies for local development/runtime setup.
- [`scripts/update_backend_versions.sh`](../scripts/update_backend_versions.sh):
  helper that copies backend-version files from sibling or fetched repos when
  setup scripts need them.

These scripts may read files such as `versions.neuriplo.env` for backend setup,
but those files are not the source of truth for which sibling repo refs the main
CMake build targets.

## Recommended Workflow

1. Use the repo's canonical configure/build/test commands for cross-repo
   maintenance.
2. Run setup scripts only when a backend dependency is needed locally.
3. Use [`cmake/versions.cmake`](../cmake/versions.cmake) to understand app-local
   sibling ref selection.
4. For video-backend setup specifics, consult `videocapture` documentation
   instead of duplicating those instructions here.

## Platform Notes

Linux is the primary supported environment for this repo and its helper scripts.
Other platforms may work for subsets of the stack, but backend support should be
verified in the owning dependency repo. See
[Deployment.md § Platform support](Deployment.md#platform-support) for the tested
matrix.
