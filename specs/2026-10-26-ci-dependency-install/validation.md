# CI dependency installation stabilization

## Requirement and plan

The v0.11.2 candidate hit the 30-minute KServe job limit during apt dependency
installation, before compilation. Bound HTTP/HTTPS idle network waits and retry
transient acquisition failures. Install optional gRPC packages with the shared
packages so the gRPC matrix entry updates package indexes only once. Preserve
all packages, build flags, tests, and the existing job time limit.

## Validation defined before implementation

- Parse both modified YAML files.
- Exercise the composite action shell with a fake sudo/apt-get command, with
  gRPC disabled and enabled. Assert one update and one install, acquisition
  timeout/retry options on both calls, unchanged common packages, and optional
  gRPC packages only when enabled.
- Run shell syntax validation and git diff --check.
- Real GitHub runner timing remains a CI check; no speedup factor is claimed
  from the command harness. Timeouts bound idle network operations, not total
  installation time or all runner failure modes.

## Results (2026-10-26)

YAML parsing and bash syntax passed. The command harness passed with gRPC
disabled and enabled: exactly one index update and one install, all original
packages retained, optional gRPC packages selected correctly, and all three
acquisition options present on both invocations. `git diff --check` passed.
Real-runner timings and full matrix validation are pending CI.
