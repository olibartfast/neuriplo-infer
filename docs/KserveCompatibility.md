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

> **TL;DR — do not serve `ecdet` as TFLite.** It loads and runs end-to-end but
> the detections are numerically garbage. The fault is the ONNX→TFLite
> conversion, not the runtime, the custom ops, or the serving path.

`ecdet_s` is an RT-DETR/D-FINE-style detector (deformable attention +
transformer: ~38 LayerNorm, ~148 MatMul, ~115 Transpose). Converting it with
[`onnx2tf`](https://github.com/PINTO0309/onnx2tf) (`flatbuffer_direct` backend,
the default) produces a `.tflite` that **runs but is numerically wrong**. To
even get it to run, the `litert` backend had to gain two kernels (see
`neuriplo/backends/litert/src/LiteRTInfer.cpp`):

- a custom `ONNX_GRIDSAMPLE` kernel (onnx2tf's lowering of ONNX `GridSample`;
  NCHW, bilinear / `zeros` / `align_corners=false`), and
- a `SIGN` builtin override that also accepts INT64 (the stock TFLite kernel is
  float-only; onnx2tf's own evaluator dies on the same node).

The `GridSample` kernel was validated against onnxruntime to `~2e-5`, so it is
**not** the problem. The problem is conversion fidelity in the transformer body
(onnx2tf's static shapes in the deformable region are even internally
inconsistent; the alternative `tf_converter` backend crashes outright on
NCHW→NHWC layout bugs).

Verified end-to-end through the C++ `neuriplo-infer` client (same pre/post-
processing, identical `data/dog.jpg` input), conf threshold 0.5:

| Backend (C++ e2e) | Top detections |
|-------------------|----------------|
| TensorRT (engine) / ONNX (onnxruntime) | bicycle 0.93, dog 0.92, car 0.74 ✅ |
| KServe `litert` (this `.tflite`)        | top score ≈0.19, random labels, nonsense boxes ❌ |

**Recommendation:** serve `ecdet` via ONNX/TensorRT/OpenVINO, not TFLite. If
TFLite is required, the model must be re-exported in a TFLite-friendly form (or
converted with a fixed onnx2tf) — there is nothing to fix in this repo. Simpler
CNN detectors (e.g. YOLO26) convert and serve correctly over `litert`; the
issue is specific to the deformable/transformer architecture.

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

# Live round-trip against both servers (needs docker + curl + grpcurl + onnx):
bash app/test/kserve_integration.sh --live

# Narrow it down:
bash app/test/kserve_integration.sh --live --servers triton --transports http

# Live client <-> neuriplo-kserve-runtime (sibling repo; HTTP + gRPC):
# ../neuriplo-kserve-client/scripts/runtime_conformance.sh --live
```

See [docs/KserveRuntime.md](KserveRuntime.md) for the full KServe runtime
reference (architecture, capabilities, configuration, build modes).
