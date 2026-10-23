# Local Inference Smoke Test

> Coverage owner: [docs/BackendCoverage.md](../../docs/BackendCoverage.md#integration-test-evidence) — this README documents the runner; coverage claims live there.

Validation status: Draft executable smoke, not a release E2E

Version set: `initial-architecture-baseline`

Owning repo: `neuriplo-infer` (this repo) — app-owned E2E runner and CLI
behavior. Task, backend, runtime, and video behavior belong to
`neuriplo-tasks`, `neuriplo`, `neuriplo-kserve-runtime`, and `videocapture`;
this smoke only exercises the runner that wires them together.

## Purpose

Validate that the local checkout can satisfy the app-owned E2E runner path
before running expensive model export or inference work.

## What It Checks

- `docker_run_inference_e2e_example.sh` exists.
- the app-owned runner can list presets.
- the app-owned runner can dry-run one representative preset.

## What It Does Not Check Yet

- model download
- model export
- backend dependency installation
- real inference execution
- numerical or visual output correctness

Use `integration-tests/kserve-runtime-yolo-e2e/run.py` for the real local KServe runtime release
validation path.

## Integration Test Plan

The release-grade local inference test should extend this smoke in ordered
stages:

1. Read the sibling pins from `versions.env`.
2. Verify each sibling checkout exists and resolves to its pinned tag.
3. Configure and build the minimum targets needed for `neuriplo-tasks`,
   `neuriplo`, and `neuriplo-infer`.
4. Run a mock or tiny fixture inference through the app-owned runner:
   `local image -> task -> backend -> result`.
5. Assert contract-level output:
   task type resolved, backend abstraction used, typed result family returned,
   and app output artifact created.
6. Record validation evidence in the PR using `ops/PR_EVIDENCE_TEMPLATE.md`.

The first executable version should prefer a mock backend or tiny fixture model
so it can run before GPU, model download, or large artifact requirements are
introduced.

## Usage

From `neuriplo-infer`:

```bash
integration-tests/local-inference-smoke/run.py
```

Override the dry-run preset:

```bash
integration-tests/local-inference-smoke/run.py --preset owlv2
```
