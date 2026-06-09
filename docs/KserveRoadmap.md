# KServe Runtime Client — Production Roadmap

Status of the `feature/neuriplo-kserve-runtime` work and the plan to bring the
KServe V2 (Open Inference Protocol) client to production quality and broad server
compatibility (KServe, Triton Inference Server, OpenVINO Model Server, TorchServe).

## Background

> **Repository split (v0.1.0):** the pure protocol client now lives in the
> standalone sibling repo
> [`neuriplo-kserve-client`](https://github.com/olibartfast/neuriplo-kserve-client)
> and is consumed here via `FetchContent` (pinned by
> `NEURIPLO_KSERVE_CLIENT_VERSION` in `versions.env`). The file paths below
> prefixed `kserve-client:` live in that repo; only the `KserveEngine` adapter
> (`app/src/KserveEngine.cpp`) and its test remain in neuriplo-infer.

The client implements the KServe V2 / Open Inference Protocol, structured (like
Triton's client library) as a **pure protocol client + an adapter** so the
client depends only on the wire protocol, never on neuriplo:

- Neutral contract: `kserve-client: include/KserveTypes.hpp` — `ModelMetadata` /
  `InferInput` / `InferOutput` (raw little-endian byte payloads) and the abstract
  `kserve::IClient`. Standard-library only, no neuriplo types.
- HTTP client: `kserve::HttpClient` (`kserve-client: src/KserveHttpClient.cpp`) —
  hand-rolled socket client.
- gRPC client: `kserve::GrpcClient` (`kserve-client: src/KserveGrpcClient.cpp`)
  over the standard `inference.GRPCInferenceService`
  (`kserve-client: proto/kserve_grpc.proto`); reads both typed `contents` and
  `raw_output_contents` (Triton's default form).
- Shared helpers: `kserve-client: KserveProtocol.{hpp,cpp}` — URL/HTTP parsing,
  datatype byte widths, and tensor `encode`/`decode` between raw bytes and
  KServe JSON.
- Adapter: `KserveEngine` (`app/src/KserveEngine.cpp`) — the **only** KServe file
  in neuriplo-infer that touches the neuriplo contract (`InferenceInterface` /
  `TensorElement` / `InferenceMetadata`). It wraps a `kserve::IClient`, caches
  metadata, and converts raw protocol bytes to typed `TensorElement`s, so a
  remote model is a drop-in for a local engine in `InferencePipeline`.

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
| 1 | ~~HTTP: `https://` stripped but connection is plaintext (no TLS)~~ — **done (Phase 3)**: OpenSSL TLS with cert verification + SNI | High | 3 |
| 2 | HTTP: response read assumes `Connection: close`; no `Content-Length` / chunked handling | High | 1 |
| 3 | HTTP: `gethostbyname` (deprecated, not thread-safe, IPv4-only) | Medium | 1 |
| 4 | Both: input datatype hardcoded `FP32`; metadata datatype ignored | High | 1 |
| 5 | Both: partial output datatype coverage | Medium | 1 |
| 6 | Both: no auth (bearer token / header / gRPC metadata) | High | 1/3 |
| 7 | ~~gRPC: `InsecureChannelCredentials` only~~ — **done (Phase 3)**: `SslCredentials` from CA/cert/key (mTLS) for `grpcs://` | High | 3 |
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

- [x] KServe binary tensor extension for HTTP and gRPC raw contents. Pure
      framing helpers live in `KserveProtocol` (`appendBinaryInput` /
      `requestBinaryOutput` / `binaryDataSize` / `splitBinaryBody` / `sliceBlob`
      plus `Inference-Header-Content-Length` parsing) and are unit-tested. HTTP
      sends inputs as raw little-endian bytes after the JSON header with
      `parameters.binary_data_size`, requests binary outputs (by name, learned
      from `modelMetadata()`), and parses binary responses symmetrically;
      gated by `KSERVE_BINARY` (opt-in, JSON stays the default/fallback). gRPC
      uses `raw_input_contents` by default (`KSERVE_BINARY=0` falls back to typed
      `contents`); the server's `raw_output_contents` were already read. Raw
      framing also widens datatype coverage (e.g. FP16/BF16) on gRPC.
- [x] Retry with exponential backoff + jitter on transient gRPC/HTTP errors.
      New pure `KserveRetry` module: a `RetryPolicy` (sourced from
      `KSERVE_MAX_RETRIES` / `KSERVE_RETRY_BASE_MS` / `KSERVE_RETRY_MAX_MS` /
      `KSERVE_RETRY_JITTER`), a deterministic `backoffDelayMs(policy, attempt,
      rand_unit)` schedule (exponential, capped, jittered), and retryable-status
      predicates for HTTP (429/502/503/504) and gRPC (UNAVAILABLE /
      DEADLINE_EXCEEDED / RESOURCE_EXHAUSTED). The `runWithRetry` loop (injectable
      sleeper + RNG, so it is unit-tested) wraps the HTTP round-trip and every
      gRPC stub call; non-retryable and final failures still throw the same
      exception types/messages as before.
- [x] Connection reuse / keep-alive. The HTTP client holds a persistent socket
      (`HttpConnection`), sends `Connection: keep-alive`, and reads exactly one
      framed response per request (honouring Content-Length / chunked instead of
      relying on EOF), reusing the socket across requests. The socket is dropped
      and transparently reconnected (via the retry loop) on any I/O error, on an
      explicit `Connection: close`, or on a server-side EOF, so stale keep-alive
      sockets degrade gracefully. Not thread-safe: calls on one client are
      assumed serialized (unchanged from before).
- [x] Server-readiness / live probes. `IClient` exposes `serverLive()` /
      `serverReady()` / `modelReady()`, implemented over HTTP
      (`/v2/health/live`, `/v2/health/ready`, `/v2/models/{m}[/versions/{v}]/ready`)
      and gRPC (`ServerLive` / `ServerReady` / `ModelReady`). `KserveEngine`
      probes `modelReady()` once before loading metadata to fail fast with a
      clear message when the server is up but the model is not ready; a transport
      failure still surfaces as a connection error. Path builders are pure
      (`KserveProtocol`) and unit-tested.

### Phase 3 — Security hardening

- [x] HTTPS for the HTTP client (OpenSSL) — new **optional** dependency. The
      socket is wrapped in a small `Connection` abstraction
      (`KserveHttpClient.cpp`) with plaintext and TLS implementations, so the
      request-building code is transport-agnostic. For an `https://` endpoint
      (`Endpoint::tls`) the TLS connection verifies the server certificate
      against the system CA roots (or a PEM CA file named by `KSERVE_CA_CERT`),
      sends SNI, and checks the certificate hostname. OpenSSL is gated behind the
      `NEURIPLO_INFER_ENABLE_KSERVE_TLS` CMake option (default ON when OpenSSL is
      found) and the `NEURIPLO_INFER_WITH_KSERVE_TLS` compile define; a build
      without OpenSSL still compiles and an `https://` endpoint then fails fast
      with a clear "built without TLS support" error.
- [x] gRPC `SslChannelCredentials` from CA/cert/key or system roots when the
      endpoint is `grpcs://` / `https://`. `KserveGrpcClient` builds
      `SslCredentialsOptions` from `KSERVE_CA_CERT` (pem_root_certs),
      `KSERVE_CLIENT_CERT` (pem_cert_chain) and `KSERVE_CLIENT_KEY`
      (pem_private_key); empty values fall back to system roots / no client cert
      (the previous behaviour).
- [x] mTLS support; secret sourcing from env/file rather than CLI. When both
      `KSERVE_CLIENT_CERT` and `KSERVE_CLIENT_KEY` are provided the gRPC channel
      presents a client certificate (mTLS); providing only one fails fast. The
      bearer token is sourced from `KSERVE_BEARER_TOKEN`, falling back to the
      file named by `KSERVE_BEARER_TOKEN_FILE` (trailing whitespace trimmed) —
      secrets never come from the command line. The pure resolution helpers
      (`trimTrailingWhitespace`, `resolveSecret`, `requireClientCertPair`) live
      in `KserveProtocol` and are unit-tested.

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

### Phase 5 — Model management (Model Repository extension)

- [x] Load / unload / repository-index over the KServe V2 Model Repository
      extension on both transports. `IClient` gains `repositoryIndex()` /
      `loadModel(name)` / `unloadModel(name)` as an **optional** capability
      (default implementations throw, so a client or test double that does not
      support model management need not override them). HTTP POSTs
      `/v2/repository/index`, `/v2/repository/models/{m}/load` and
      `/v2/repository/models/{m}/unload`; gRPC calls `RepositoryIndex` /
      `RepositoryModelLoad` / `RepositoryModelUnload` (added to
      `proto/kserve_grpc.proto` with field numbers matching the official
      KServe/Triton service for wire compatibility). The neutral
      `RepositoryModel` result and the pure path builders + `parseRepositoryIndex`
      helper live in `KserveTypes` / `KserveProtocol` and are unit-tested; the
      repository calls reuse the existing retry/backoff and auth/TLS plumbing.
      load/unload take an explicit model name (independent of the client's bound
      inference model) and throw on a non-success response.

## Out of scope (for now)

- Streaming inference.
- Server-side batching configuration.
- CLI exposure of the model-management API (the capability lives on the client;
  no admin subcommand is wired into the inference CLI yet).
