# Requirements — NERT default and CI

Branch: `feature/nert-default-docs`
Roadmap: maintainer-requested prerequisite to the next release.

## Goal

Make NERT the application default, document it consistently, and build/test it
in the existing fail-fast CI matrix before releasing.

## Scope and decisions

- The maintainer identified NERT as the default and requested README/docs
  corrections and NERT CI for neuriplo-infer.
- Fresh builds select CPU NERT through the pinned neuriplo v0.11.3 factory.
  Explicit backend selections and existing CMake caches retain their selection.
- Dependency setup defaults to NERT without installing a separate SDK: neuriplo
  fetches the pinned engine. OpenCV remains an app dependency: 4.6 minimum,
  source-built 5.x preferred when available.
- Keep the explicit OPENCV_DNN CI job; add a NERT job that omits the backend
  override, runs app tests, and verifies the advertised local backend.
- Preserve task contracts, CLI flags, schema version, fallback behavior, and
  inference implementation. No upstream or engine changes, CUDA enablement,
  model-coverage claims, Docker image, or dependency pin moves.

## CI installation correction

Hosted job logs show Azure's Ubuntu HTTP archive downloading at about 40 KB/s,
exhausting the 30-minute job budget before compilation. The shared install
action switches that archive URL to the official Ubuntu HTTPS archive while
preserving package names, version selection, and signed indexes. Cover both
direct apt sources and the hosted runner’s `/etc/apt/apt-mirrors.txt` indirection.

## Release gate

The maintainer requires master and develop CI to be green before release.
The new default must be validated before tagging or publishing.

## Open questions

None for this bounded CPU wiring and CI change; GPU policy remains upstream.
