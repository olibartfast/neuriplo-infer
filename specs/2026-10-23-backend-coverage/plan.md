# Feature Plan — Backend coverage matrix

Derived from `requirements.md`.

## Group 1 — Foundation (evidence census)

1. Confirm backend list and artifact formats against `docs/Deployment.md`
   backends table and `neuriplo` backend inventory (9 local + KServe only).
2. Confirm task rows against `docs/generated/supported-model-types.md` and
   `app/src/Capabilities.cpp` `taskCapabilities()` (10 families).
3. Census existing evidence per cell: `docker_run_inference_e2e_example.sh`
   preset defaults, `.github/workflows/e2e-example-emulation.yml` matrix,
   `.github/workflows/ci.yml` build jobs, `docs/KserveCompatibility.md`
   validated cells. Record level per preset (live / builds / dry-run-only).

## Group 2 — Behavior (matrix doc)

4. Write `docs/BackendCoverage.md`: Definitions, backend capability table,
   task × backend gate table with Evidence column, scoreboard, open items,
   non-kernel limits, sources. Every `PASS` cites its recorded output;
   everything else cites build/preset/compat source or is `UNTESTED`.

## Group 3 — Contract and documentation

5. Link the matrix from `README.md` docs table and `docs/Deployment.md`
   backends section (one line each, no prose duplication).
6. Add `CHANGELOG.md` `[Unreleased]` entry. No `--capabilities`, schema, or
   generated-docs changes.

## Group 4 — Verification

7. Run `validation.md`: default + test builds, full `ctest`, generated-docs
   `--check`, preset dry-runs for the matrix's representative presets,
   hyperlink checks for every added link.
