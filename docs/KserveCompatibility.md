# KServe Server Compatibility Matrix

The KServe client speaks the KServe V2 / Open Inference Protocol (OIP), so it is
wire-compatible in principle with any server that implements it. This document
records the combinations that are **actually exercised by CI** and how, so the
matrix stays honest rather than aspirational.

> The protocol client itself lives in the standalone
> [`neuriplo-kserve-client`](https://github.com/olibartfast/neuriplo-kserve-client)
> repo (fetched here via `FetchContent`); neuriplo-infer keeps the `KserveEngine`
> adapter and this integration harness.

It is kept green by the integration harness
[`app/test/kserve_integration.sh`](../app/test/kserve_integration.sh) and the
[`KServe Integration`](../.github/workflows/kserve-integration.yml) workflow:

- **Dry-run** (every push/PR, no images/GPU): validates that the client driver
  and server launch commands are well-formed for each server × transport. Also
  registered with CTest as `kserve_integration_dry_run` (runs by default).
- **Live** (manual `workflow_dispatch` with `run_live=true`): starts Triton and
  OVMS containers serving a tiny shared ONNX `Identity` model and asserts a
  successful KServe V2 inference round-trip over HTTP and gRPC against each.

## Server × transport

Legend: ✅ exercised live in CI · 🟡 dry-run only (live behind manual dispatch)
· ⏳ not yet covered.

| Server | HTTP | gRPC | Notes |
|--------|------|------|-------|
| neuriplo-kserve-runtime | ✅ validated | ✅ validated | reference target; gRPC live-validated 2026-06-12 via client conformance + infer `KserveEngine` after runtime PR #9 (`raw_output_contents`) |
| Triton Inference Server | 🟡 (live: dispatch) | 🟡 (live: dispatch) | ONNX backend; `grpcurl` over `proto/kserve_grpc.proto` |
| OpenVINO Model Server (OVMS) | 🟡 (live: dispatch) | 🟡 (live: dispatch) | same OIP endpoints; serves ONNX directly |
| TorchServe (OIP) | ⏳ | ⏳ | version routing varies |
| Custom TPU server (operator-supplied, KServe V2) | ⏳ (dry-run only) | ⏳ | client scaffold validated, not live-validated; HTTP is the first path, gRPC applies only if the operator service implements it; see the [TPU harness](../integration-tests/tpu-kserve/README.md) and [BackendCoverage E18](BackendCoverage.md#ev-accelerator) |

"Live: dispatch" means the round-trip is implemented and run by the live job,
which is gated behind a manual dispatch so routine CI never pulls multi-GB
server images. The dry-run path for these servers runs on every PR.

### Model formats

The reference `neuriplo-kserve-runtime` delegates execution to neuriplo
backends, so it serves whatever format the backend it was built with consumes
(ONNX, OpenVINO IR, TorchScript, TFLite, ...). It reports the serving backend
as `platform: neuriplo_<backend>`, which `KserveEngine` surfaces and the CLI
folds into the output filename (`processed_<model>_kserve_<backend>.png`).
**TFLite** over the `litert` backend was validated locally on 2026-06-13
(runtime + `neuriplo-infer` client round-trip on a `.tflite` model — YOLO26,
which serves correctly); it is not yet wired into the CI live job. Triton and
OVMS in the matrix above serve ONNX.

#### Known-bad conversion: EdgeCrafter `ecdet` (RT-DETR/deformable) → TFLite

The legacy converter artifact is unsupported. The supported replacement
passes local C++ and KServe HTTP checks (2026-11-04):

1. Reconvert the ONNX model with `onnx2tf 2.6.9` (pure builtin operators).
2. Use the neuriplo LiteRT backend patch enabling XNNPACK subgraph reshaping.
   The default TF 2.19 delegate cannot handle the dynamic graph; its builtin
   execution produces deflated scores. This patch is not yet released
   or included in the app's dependency pins.
3. Re-gate against the original ONNX model before deploying a new artifact.

With identical app inputs, ORT and patched C++ LiteRT return dog 0.929990,
bicycle 0.927981 and car 0.780462; the maximum score difference over all
300 queries is 4.49e-6. KServe HTTP returns tensors identical to local LiteRT.
Legacy TFLite artifacts are unsupported. gRPC status remains unvalidated.

See [BackendCoverage E4](BackendCoverage.md#ev-ecdet) and the exact
[validation](../specs/2026-11-04-known-bad-backends/validation.md). RF-DETR
detection and segmentation use their native LiteRT exporter instead of the
known-bad generic conversion route; the app presets select that exporter
with `--backend litert`.

## Datatype coverage

Input datatypes are taken from the server's model metadata (not hardcoded). The
adapter ([`KserveEngine`](../app/src/KserveEngine.cpp)) decodes outputs into the
`TensorElement` variant (`float` / `int32` / `int64` / `uint8`), widening or
narrowing wider server datatypes.

| KServe datatype | HTTP | gRPC | Decoded as |
|-----------------|------|------|------------|
| FP32 | ✅ | ✅ | float |
| FP64 | ✅ | ✅ | float (narrowed) |
| FP16 | ✅ | ✅ | float (gRPC via raw tensor contents — the default; the typed-`contents` fallback `KSERVE_BINARY=0` cannot carry it) |
| INT8 / INT16 / INT32 | ✅ | ✅ | int32 |
| INT64 | ✅ | ✅ | int64 |
| UINT8 / BOOL | ✅ | ✅ | uint8 |
| UINT16 / UINT32 / UINT64 | ✅ | ✅ | int64 (widened) |

The integration harness round-trips **FP32** end-to-end (the datatype shared by
the tiny model on both servers); the wider datatype decoding is covered by the
`KserveProtocol` / `KserveEngine` unit tests.

Image **inputs** are a narrower contract: preprocessing produces `FP32` or raw
`UINT8` pixels, so an image input advertised as anything else is refused at
pipeline setup, and every input's byte count is checked against its advertised
datatype before a request is sent. `app/test/kserve_input_datatypes_e2e.sh`
proves this against `neuriplo-kserve-runtime` — `UINT8` and `FP32` image inputs
run, `INT8` and `BOOL` are refused with no request reaching the server. CTest
runs it as `kserve_input_datatypes_e2e_dry_run`; `--live` needs the runtime and
`neuriplo-infer` binaries.

## Authentication & transport security

- Bearer token via `KSERVE_BEARER_TOKEN` (HTTP `Authorization: Bearer …`, gRPC
  call metadata).
- TLS selection by scheme (`https://` / `grpcs://`); HTTPS requires an
  OpenSSL-enabled build. Optional mTLS via `KSERVE_CLIENT_CERT` /
  `KSERVE_CLIENT_KEY`; the CA bundle comes from `KSERVE_CA_CERT` (system roots
  when unset).

## Model management (Model Repository extension)

The client implements the KServe V2 Model Repository extension on both
transports — `repositoryIndex()` / `loadModel(name)` / `unloadModel(name)`
(HTTP `POST /v2/repository/index|models/{m}/load|models/{m}/unload`; gRPC
`RepositoryIndex` / `RepositoryModelLoad` / `RepositoryModelUnload`). This is an
optional server capability: Triton exposes it (with
`--model-control-mode=explicit`), and KServe/OVMS support varies by deployment,
so it is **not** part of the routine integration round-trip. The pure path
builders and index parser are unit-tested; the wire calls reuse the existing
retry/auth/TLS plumbing. See [docs/KserveRuntime.md](KserveRuntime.md).

## Running it yourself

```bash
# Dry-run (no docker needed) — what CI runs on every PR:
bash app/test/kserve_integration.sh --dry-run

# Live round-trip against both servers (needs docker + curl + grpcurl + onnx).
# --require-live fails if prerequisites are missing instead of skipping:
bash app/test/kserve_integration.sh --live --require-live

# Narrow it down:
bash app/test/kserve_integration.sh --live --servers triton --transports http

# Live client <-> neuriplo-kserve-runtime (sibling repo; HTTP + gRPC):
# ../neuriplo-kserve-client/scripts/runtime_conformance.sh --live
```

See [docs/KserveRuntime.md](KserveRuntime.md) for the full KServe runtime
reference (architecture, capabilities, configuration, build modes).
