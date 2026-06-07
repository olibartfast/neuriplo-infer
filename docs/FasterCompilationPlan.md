# Faster C++ Compilation Plan

This plan keeps build-speed work small, measurable, and separate from inference
behavior. It applies to local development builds for `neuriplo-infer`.

## Goals

- Reduce clean and incremental compile time.
- Preserve CLI compatibility, output schema, backend fallback behavior, and
  inference semantics.
- Avoid new mandatory runtime dependencies.
- Keep each improvement independently reviewable.

## Baseline First

Measure before changing build settings.

```bash
time cmake --build build-test
```

Record:

- generator: Makefiles or Ninja
- compiler and version
- CPU core count
- clean build time
- one-file incremental build time
- link time, if visibly slow
- `ccache -s`, if `ccache` is already installed

## Phase 1: Low-Risk Local Speedups

Use Ninja, parallel jobs, and optional compiler cache. These do not change source
code or runtime behavior.

```bash
cmake -S . -B build-fast -G Ninja \
  -DDEFAULT_BACKEND=OPENCV_DNN \
  -DENABLE_APP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build build-fast -j"$(nproc)"
ctest --test-dir build-fast --output-on-failure
```

Acceptance:

- build succeeds
- tests pass
- incremental rebuild improves after cache warmup

Notes:

- If `ccache` is unavailable, omit `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`.
- Keep canonical repo commands in `ops/repo-meta/neuriplo-infer.yaml`
  unchanged unless CI or maintainers explicitly adopt the faster profile.

## Phase 2: Faster Linker When Link Time Dominates

Use only when measurements show linking is a meaningful part of build time.

Preferred linker:

```bash
cmake -S . -B build-fast -G Ninja \
  -DDEFAULT_BACKEND=OPENCV_DNN \
  -DENABLE_APP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=mold"
```

Fallback linker:

```bash
cmake -S . -B build-fast -G Ninja \
  -DDEFAULT_BACKEND=OPENCV_DNN \
  -DENABLE_APP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
```

Acceptance:

- configure succeeds on developer machines where linker is installed
- build succeeds
- tests pass
- linker choice remains optional, not required for default builds

## Phase 3: Include Hygiene

Use when incremental builds remain slow because common headers trigger broad
rebuilds.

Actions:

- replace unnecessary includes in headers with forward declarations
- move heavy includes from headers to `.cpp` files
- keep public headers stable and minimal
- avoid touching cross-repo contracts unless required

Acceptance:

- no public API behavior changes
- build succeeds
- tests pass
- compile database or rebuild logs show smaller rebuild fanout

## Phase 4: Precompiled Headers

Use only after include hygiene and measurements justify it.

Candidate headers:

- stable STL headers
- stable OpenCV headers used widely in app sources
- stable project headers with low churn

Avoid:

- headers that frequently change
- generated headers
- backend-specific headers that make builds less portable

Acceptance:

- clean build improves or stays acceptable
- incremental build does not regress
- PCH setup is target-scoped and easy to disable
- tests pass

## Phase 5: Unity Builds

Use as an experiment for clean-build speed, not as default policy without review.

Risks:

- hidden include-order problems
- duplicate symbols or anonymous namespace collisions
- different warning behavior
- less useful per-file diagnostics

Acceptance:

- clean build improves enough to justify risk
- no new warnings or ODR issues
- tests pass
- unity mode remains opt-in unless maintainers approve default use

## Phase 6: Test Build Scope

Use when local loops spend most time rebuilding or relinking tests.

Actions:

- identify slowest test targets
- keep focused local targets for changed areas
- preserve full `ctest --test-dir build-test --output-on-failure` before PR
  evidence

Acceptance:

- fast local loop exists for common edits
- full repo-local validation remains documented and runnable
- no test coverage removed

## Recommended Order

1. Measure current build.
2. Adopt `build-fast` with Ninja and parallel jobs.
3. Add `ccache` when available.
4. Try `mold` or `lld` only if link time is high.
5. Improve includes only where rebuild fanout is measured.
6. Consider PCH, unity builds, or test-scope changes only after data shows
   earlier phases are insufficient.

## PR Evidence

For any committed build-speed change, include:

- before and after build timings
- configure command
- build command
- test command and result
- note confirming no inference semantics, output schema, CLI flags, or backend
  fallback behavior changed
