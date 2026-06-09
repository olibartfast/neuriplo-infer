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
| neuriplo-kserve-runtime | ✅ validated | 🟡 | reference target |
| Triton Inference Server | 🟡 (live: dispatch) | 🟡 (live: dispatch) | ONNX backend; `grpcurl` over `proto/kserve_grpc.proto` |
| OpenVINO Model Server (OVMS) | 🟡 (live: dispatch) | 🟡 (live: dispatch) | same OIP endpoints; serves ONNX directly |
| TorchServe (OIP) | ⏳ | ⏳ | version routing varies |

"Live: dispatch" means the round-trip is implemented and run by the live job,
which is gated behind a manual dispatch so routine CI never pulls multi-GB
server images. The dry-run path for these servers runs on every PR.

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
retry/auth/TLS plumbing. See [docs/KserveRoadmap.md](KserveRoadmap.md) Phase 5.

## Running it yourself

```bash
# Dry-run (no docker needed) — what CI runs on every PR:
bash app/test/kserve_integration.sh --dry-run

# Live round-trip against both servers (needs docker + curl + grpcurl + onnx):
bash app/test/kserve_integration.sh --live

# Narrow it down:
bash app/test/kserve_integration.sh --live --servers triton --transports http
```

See [docs/KserveRoadmap.md](KserveRoadmap.md) for the completed production
roadmap (kept as the design record).
