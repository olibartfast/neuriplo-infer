# KServe Runtime Client — Production Roadmap

Status of the `feature/neuriplo-kserve-runtime` work and the plan to bring the
KServe V2 (Open Inference Protocol) client to production quality and broad server
compatibility (KServe, Triton Inference Server, OpenVINO Model Server, TorchServe).

## Background

The client implements the KServe V2 / Open Inference Protocol, structured (like
Triton's client library) as a **pure protocol client + an adapter** so the
client depends only on the wire protocol, never on neuriplo:

- Neutral contract: `KserveTypes.hpp` — `ModelMetadata` / `InferInput` /
  `InferOutput` (raw little-endian byte payloads) and the abstract
  `kserve::IClient`. Standard-library only, no neuriplo types.
- HTTP client: `kserve::HttpClient` (`app/src/KserveHttpClient.cpp`) — hand-rolled
  socket client.
- gRPC client: `kserve::GrpcClient` (`app/src/KserveGrpcClient.cpp`) over the
  standard `inference.GRPCInferenceService` (`proto/kserve_grpc.proto`); reads
  both typed `contents` and `raw_output_contents` (Triton's default form).
- Shared helpers: `KserveProtocol.{hpp,cpp}` — URL/HTTP parsing, datatype byte
  widths, and tensor `encode`/`decode` between raw bytes and KServe JSON.
- Adapter: `KserveEngine` (`app/src/KserveEngine.cpp`) — the **only** KServe file
  that touches the neuriplo contract (`InferenceInterface` / `TensorElement` /
  `InferenceMetadata`). It wraps a `kserve::IClient`, caches metadata, and
  converts raw protocol bytes to typed `TensorElement`s, so a remote model is a
  drop-in for a local engine in `InferencePipeline`.

Because the protocol is the same one Triton and OpenVINO Model Server expose, the
client is wire-compatible with all of them **in principle**; the work below closes
the practical gaps.

## Constraints discovered

- `InferenceMetadata` / `LayerInfo` come from the external `neuriplo` backend
  library and carry no datatype field. Datatypes must therefore be captured and
  held inside the KServe clients.
- `TensorElement` is `std::variant<float, int32_t, int64_t, uint8_t>`. Output
  decoding is bounded to those four C++ types; wider server datatypes (FP16, BOOL,
  INT8/16, UINT16/32/64) are widened/narrowed into them.

## Gap analysis

### Point 1 — Production readiness

| # | Gap | Severity | Phase |
|---|-----|----------|-------|
| 1 | HTTP: `https://` stripped but connection is plaintext (no TLS) | High | 3 |
| 2 | HTTP: response read assumes `Connection: close`; no `Content-Length` / chunked handling | High | 1 |
| 3 | HTTP: `gethostbyname` (deprecated, not thread-safe, IPv4-only) | Medium | 1 |
| 4 | Both: input datatype hardcoded `FP32`; metadata datatype ignored | High | 1 |
| 5 | Both: partial output datatype coverage | Medium | 1 |
| 6 | Both: no auth (bearer token / header / gRPC metadata) | High | 1/3 |
| 7 | gRPC: `InsecureChannelCredentials` only | High | 3 |
| 8 | No retry / backoff on transient failures | Medium | 2 |
| 9 | No binary tensor extension (JSON float arrays only) | Medium (perf) | 2 |
| 10 | No inference round-trip tests | High | 1 |

### Point 2 — Ready to close on develop

Exit criteria to merge `feature/neuriplo-kserve-runtime` into `develop`:

- [x] Default transport points at the validated path (`http`).
- [x] Datatype no longer hardcoded; driven by fetched metadata (HTTP + gRPC).
- [x] HTTP responses parsed correctly (Content-Length + chunked).
- [x] Auth supported on both paths (`KSERVE_BEARER_TOKEN`).
- [x] Unit tests for protocol helpers added and passing.
- [x] CI matrix builds with and without gRPC / with `ENABLE_KSERVE=OFF` and a
      `kserve-only` (LOCAL_BACKENDS=OFF) config; asserts KServe sources absent in
      local-only and neuriplo not fetched in kserve-only (`ci.yml` `kserve-build-matrix`).
- [x] README/Known Limitations updated to match actual capability.

### Point 3 — Works with any KServe server

Compatibility matrix (target after Phase 1):

| Server | HTTP | gRPC | Notes |
|--------|------|------|-------|
| neuriplo-kserve-runtime | ✅ validated | ⏳ | reference target |
| Triton Inference Server | ✅ (FP32/UINT8/INT) | ✅ | binary ext (Phase 2) for perf |
| OpenVINO Model Server | ✅ | ✅ | same OIP endpoints |
| TorchServe (OIP) | ✅ | ✅ | version routing varies |

Blockers to "any server" are items 1, 4, 5, 6, 7 above — all addressed across
Phases 1 and 3.

## Phased plan

### Phase 1 — Correctness & compatibility (this iteration)

1. Extract pure helpers into `KserveProtocol.{hpp,cpp}` (URL parsing, HTTP
   response parsing incl. chunked de-chunking, datatype byte-width + encode/decode).
2. Drive datatype from metadata in both clients; widen output decoding.
3. `getaddrinfo`, `Content-Length`/chunked handling in HTTP client.
4. Bearer-token / header auth (HTTP) and call-credential metadata (gRPC).
5. Flip default transport to `http` (validated); keep gRPC opt-in.
6. Unit tests for the protocol helpers; wire into `app/test`.
7. Update README + Known Limitations.

### Phase 2 — Performance & resilience

- KServe binary tensor extension for HTTP and gRPC raw contents.
- Retry with exponential backoff + jitter on transient gRPC/HTTP errors.
- Connection reuse / keep-alive.
- [x] Server-readiness / live probes. `IClient` exposes `serverLive()` /
      `serverReady()` / `modelReady()`, implemented over HTTP
      (`/v2/health/live`, `/v2/health/ready`, `/v2/models/{m}[/versions/{v}]/ready`)
      and gRPC (`ServerLive` / `ServerReady` / `ModelReady`). `KserveEngine`
      probes `modelReady()` once before loading metadata to fail fast with a
      clear message when the server is up but the model is not ready; a transport
      failure still surfaces as a connection error. Path builders are pure
      (`KserveProtocol`) and unit-tested.

### Phase 3 — Security hardening

- HTTPS for the HTTP client (OpenSSL/BoringSSL) — new optional dependency.
- gRPC `SslChannelCredentials` from CA/cert/key or system roots when endpoint is
  `grpcs://` / `https://`.
- mTLS support; secret sourcing from env/file rather than CLI.

### Phase 4 — Productionization

- [x] Integration test against a containerized Triton + OVMS in CI (dry-run +
      live). `app/test/kserve_integration.sh` serves a tiny shared ONNX
      `Identity` model from Triton and OVMS and drives a KServe V2 round-trip
      over HTTP (curl) and gRPC (grpcurl over `proto/kserve_grpc.proto`) against
      each. Dry-run mode validates command construction and skips when
      docker/images/tooling are absent (mirrors the e2e script's gating), so CI
      without GPUs still passes; registered with CTest as
      `kserve_integration_dry_run` (KServe builds only) and run on every PR by
      the new `kserve-integration.yml` workflow. The live path is gated behind a
      manual `workflow_dispatch` (`run_live=true`) so routine CI never pulls the
      multi-GB server images.
- [x] Observability: per-request latency surfaced through `KserveEngine`.
      `get_infer_results` times the remote round-trip (around
      `IClient::infer()`) and emits a `glog VLOG(1)` debug line per request;
      callers can read `lastInferenceLatencyMs()` /
      `averageInferenceLatencyMs()` / `inferenceCount()`. Covered by
      `test_KserveEngine.cpp` (fake client). The base class already times the
      whole task pipeline; this isolates the network/inference component.
- [x] Documented compatibility matrix kept green by CI.
      `docs/KserveCompatibility.md` records the tested server/transport/datatype
      combinations and is tied to the integration harness above (dry-run on
      every PR, live behind dispatch); referenced from `README.md`.

## Build decoupling: local-only vs KServe-only

Two independent axes:

### Does local (in-process) inference still work with KServe present? — Yes

KServe is purely additive. `InferencePipelineBuilder::setupBackend`
(`app/src/InferencePipeline.cpp:185`) only constructs a KServe engine when
`--kserve_endpoint` is non-empty; otherwise it builds the local neuriplo backend.
The two are mutually exclusive **per invocation** and selected at runtime. No
client/server is required to run local inference.

### "Build only the KServe client and skip fetching neuriplo" — yes

The KServe **clients** depend on nothing from neuriplo (see Background: pure
`kserve::IClient` + protocol helpers, raw-byte payloads). Only the thin
`KserveEngine` adapter and the pipeline reference the inference contract
(`InferenceInterface` / `TensorElement` / `InferenceMetadata`). To build without
fetching neuriplo at all, that contract is provided by **app-local headers** in
`app/inc/contract/` (dependency-free; no OpenCV, no neuriplo). They are put on
the include path only in KServe-only builds, so the same source compiles
unchanged in both modes — nothing external (neuriplo *or* a core library) is
fetched.

Build-mode switches:

- `NEURIPLO_INFER_ENABLE_KSERVE` (default `ON`) — gates HTTP + gRPC clients and
  CLI plumbing. `OFF` → no KServe code, no gRPC/protobuf needed.
- `NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS` (default `ON`) — gates the local
  in-process engines. This is the **only** thing that fetches/builds `neuriplo`
  (and the heavyweight ONNX/TensorRT/LibTorch runtimes). `OFF` → no neuriplo
  fetch; the contract comes from `app/inc/contract/`; `setup_inference_engine`
  is compiled out and `--kserve_endpoint` becomes mandatory at runtime.
- `NEURIPLO_INFER_ENABLE_GRPC` (existing) — gates only the gRPC transport.

At least one of `ENABLE_KSERVE` / `ENABLE_LOCAL_BACKENDS` must be `ON` (enforced
at configure time). The build matrix:

| KSERVE | LOCAL_BACKENDS | Fetches neuriplo? | Result |
|:------:|:--------------:|:-----------------:|--------|
| ON  | ON  | yes | local engines + remote KServe |
| OFF | ON  | yes | local engines only (no KServe code) |
| ON  | OFF | **no**  | KServe-only; contract from `app/inc/contract/` |
| OFF | OFF | — | configure error |

Note: a KServe-only build still uses OpenCV and `neuriplo-tasks` (the app's image
pipeline and task layer); what it avoids is the `neuriplo` backend repo and its
per-backend runtimes. `neuriplo-tasks` does not depend on `neuriplo`.

| Build intent | neuriplo fetch | KServe code | How |
|--------------|----------------|-------------|-----|
| Local only | required | none | `-DNEURIPLO_INFER_ENABLE_KSERVE=OFF` |
| Local + KServe (default) | required | HTTP (+gRPC) | defaults |
| KServe only, no neuriplo | not needed | HTTP (+gRPC) | needs `neuriplo-core` extraction (future) |

## Out of scope (for now)

- Streaming inference.
- Model management API (load/unload).
- Server-side batching configuration.
