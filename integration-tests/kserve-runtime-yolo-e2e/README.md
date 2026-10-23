# YOLO KServe Runtime E2E Test

> Coverage owner: [docs/BackendCoverage.md](../../docs/BackendCoverage.md#integration-test-evidence) — this README documents the runner; coverage claims live there.

Validation status: Real local E2E

Version set: `feature/neuriplo-kserve-runtime` sibling branch set, with local
built binaries and model artifacts.

Owning repos:

- `neuriplo-infer`: owns this test and the app-layer KServe V2 client binary
  that calls a remote endpoint.
- `neuriplo-kserve-runtime`: KServe V2 serving runtime process.
- `neuriplo`: backend abstraction and ONNX Runtime execution.
- `neuriplo-tasks`: YOLO task preprocessing and postprocessing contracts.

## Purpose

Run a real local release-gate path for the serving runtime instead of a dry-run
smoke. The test starts `neuriplo-kserve-runtime`, waits for KServe readiness,
checks model metadata, invokes this repo's app binary against the HTTP
KServe endpoint, verifies a refreshed rendered output image, checks success and
failure metrics, and stops the runtime. This validates `neuriplo-kserve-runtime`
as one compatible server for the protocol-level client path.

## Required Local Artifacts

Default paths assume sibling checkouts next to `neuriplo-infer`:

- `../neuriplo-kserve-runtime/build/real-onnx/neuriplo-kserve-runtime`
- `build-kserve-codex/app/neuriplo-infer`
- `models/e2e/yolo26s.onnx` or `yolo26s.onnx`
- `data/dog.jpg`
- `labels/coco.names`

## Usage

From `neuriplo-infer`:

```bash
integration-tests/kserve-runtime-yolo-e2e/run.py
```

Override paths when using different build directories or model locations:

```bash
integration-tests/kserve-runtime-yolo-e2e/run.py   --runtime-bin ../neuriplo-kserve-runtime/build/real-onnx/neuriplo-kserve-runtime   --infer-build-dir build-kserve-codex   --model yolo26s.onnx
```

## What It Checks

- runtime process starts with the real `onnx_runtime` backend
- `/v2/health/live` and `/v2/health/ready` report true
- `/v2/models/yolo` reports `neuriplo_onnx_runtime`, input `images`, and output
  `output0`
- the app-layer executable sends a real KServe HTTP inference request without
  depending on `neuriplo` backend internals
- `../neuriplo-infer/data/output/processed_<model>_<backend>.png` (e.g.
  `processed_yolo26_kserve.png`) is refreshed and non-empty
- Prometheus metrics report one successful infer request and zero failures

## Current Limitations

- HTTP KServe V2 is validated; gRPC parity remains pending until the sibling
  build environment provides gRPC.
- This test requires local binaries and model artifacts; it does not download
  models or build dependencies.
