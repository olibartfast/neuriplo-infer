# KServe Runtime Mode — Reference

How `neuriplo-infer` runs inference against a remote KServe V2 / Open Inference
Protocol (OIP) server — Triton Inference Server, OpenVINO Model Server,
TorchServe (OIP), KServe, or `neuriplo-kserve-runtime` — while task
preprocessing, postprocessing, and rendering stay in this app.

Companion documents:

- [KserveCompatibility.md](KserveCompatibility.md) — the CI-backed matrix of
  tested server / transport / datatype combinations.
- `README.md` § *KServe Runtime Parameters* — CLI flags and usage examples.
- History: the feature was developed on `feature/neuriplo-kserve-runtime` and
  merged into `develop` on 2026-06-09 with all roadmap phases complete; see
  `CHANGELOG.md` and git history for the phase-by-phase record.

## Architecture

The implementation follows the same structure as Triton's client library: a
**pure protocol client + a thin adapter**, so the client depends only on the
wire protocol, never on neuriplo.

The protocol client lives in the standalone sibling repo
[`neuriplo-kserve-client`](https://github.com/olibartfast/neuriplo-kserve-client)
and is consumed via `FetchContent`, pinned by `NEURIPLO_KSERVE_CLIENT_VERSION`
in `versions.env`. Paths prefixed `kserve-client:` below live in that repo.

| Component | Location | Role |
|-----------|----------|------|
| Neutral contract | `kserve-client: include/KserveTypes.hpp` | `ModelMetadata` / `InferInput` / `InferOutput` (raw little-endian byte payloads), `RepositoryModel`, and the abstract `kserve::IClient`. Standard-library only. |
| HTTP client | `kserve-client: src/KserveHttpClient.cpp` | Hand-rolled socket client; keep-alive, chunked/Content-Length framing, optional TLS (OpenSSL), binary tensor extension. |
| gRPC client | `kserve-client: src/KserveGrpcClient.cpp` | Standard `inference.GRPCInferenceService` (`kserve-client: proto/kserve_grpc.proto`); raw and typed tensor contents, TLS/mTLS. |
| Protocol helpers | `kserve-client: KserveProtocol.{hpp,cpp}` | Pure, unit-tested: URL/HTTP parsing, datatype byte widths, tensor encode/decode, binary framing, retry policy, secret resolution. |
| Adapter | `app/src/KserveEngine.cpp` (+ `app/inc/KserveEngine.hpp`) | The **only** KServe file in neuriplo-infer touching the neuriplo contract (`InferenceInterface` / `TensorElement` / `InferenceMetadata`). Wraps a `kserve::IClient`, caches metadata, converts raw bytes to typed `TensorElement`s, and surfaces per-request latency. |

`InferencePipelineBuilder::setupBackend` (`app/src/InferencePipeline.cpp`)
constructs a KServe engine only when `--kserve_endpoint` is non-empty;
otherwise it builds a local neuriplo backend. Remote and local inference are
selected at runtime and are mutually exclusive per invocation.

## Capabilities

- **Transports**: HTTP (default, validated path) and gRPC (built when
  Protobuf/gRPC are available; `--kserve_transport=grpc`).
- **Datatypes**: input datatypes come from the server's model metadata (never
  hardcoded). Outputs decode into the `TensorElement` variant — `float` /
  `int32_t` / `int64_t` / `uint8_t` — with wider server datatypes widened or
  narrowed (see the datatype table in
  [KserveCompatibility.md](KserveCompatibility.md)). FP16/BF16 over gRPC
  require raw tensor contents (the default).
- **Binary tensors**: KServe binary tensor extension on HTTP (opt-in via
  `KSERVE_BINARY=1`; JSON is the default) and raw `raw_input_contents` /
  `raw_output_contents` on gRPC (the default; `KSERVE_BINARY=0` falls back to
  typed `contents`).
- **Resilience**: retry with exponential backoff + jitter on transient
  failures — HTTP 429/502/503/504; gRPC UNAVAILABLE / DEADLINE_EXCEEDED /
  RESOURCE_EXHAUSTED. HTTP connections are persistent (keep-alive) and
  transparently reconnected on I/O errors or server-side close. Calls on one
  client are assumed serialized (not thread-safe).
- **Health probes**: `serverLive()` / `serverReady()` / `modelReady()` on both
  transports. `KserveEngine` probes `modelReady()` before loading metadata to
  fail fast with a clear message when the server is up but the model is not.
- **Auth & TLS**: bearer token (HTTP `Authorization` header / gRPC call
  metadata); HTTPS with certificate verification + SNI (OpenSSL build);
  `grpcs://` with `SslCredentials`; optional mTLS. Secrets come from
  environment variables or files, never the command line.
- **Observability**: the adapter times each remote round-trip
  (`lastInferenceLatencyMs()` / `averageInferenceLatencyMs()` /
  `inferenceCount()`) and emits a glog `VLOG(1)` line per request.
- **Model management**: KServe V2 Model Repository extension on both
  transports — `repositoryIndex()` / `loadModel(name)` / `unloadModel(name)`
  on `IClient` as an optional capability (base methods throw). Requires server
  support (e.g. Triton `--model-control-mode=explicit`); not exposed through
  the CLI.

## Configuration

CLI flags (`--kserve_endpoint`, `--kserve_model_name`,
`--kserve_model_version`, `--kserve_transport`, `--kserve_timeout_ms`) are
documented in `README.md` § *KServe Runtime Parameters*. The endpoint scheme
selects transport security: `http://` / `grpc://` plaintext, `https://` /
`grpcs://` TLS.

Environment variables (the canonical list):

| Variable | Effect |
|----------|--------|
| `KSERVE_BEARER_TOKEN` | Bearer token, sent as `Authorization: Bearer …` (HTTP) / call metadata (gRPC). |
| `KSERVE_BEARER_TOKEN_FILE` | File holding the token, used when `KSERVE_BEARER_TOKEN` is unset (trailing whitespace trimmed). |
| `KSERVE_CA_CERT` | PEM CA bundle for verifying the server certificate (`https://` / `grpcs://`). Defaults to system roots. |
| `KSERVE_CLIENT_CERT` / `KSERVE_CLIENT_KEY` | PEM client certificate + key; both → mTLS, only one → error. |
| `KSERVE_BINARY` | Binary tensor extension. HTTP: opt-in (`1`). gRPC: raw contents are the default; `0` falls back to typed `contents` (no FP16/BF16). |
| `KSERVE_MAX_RETRIES` | Retry attempts on transient failures. |
| `KSERVE_RETRY_BASE_MS` / `KSERVE_RETRY_MAX_MS` | Exponential backoff base and cap. |
| `KSERVE_RETRY_JITTER` | Backoff jitter factor. |

## Build modes

Two independent CMake switches (plus the gRPC gate):

- `NEURIPLO_INFER_ENABLE_KSERVE` (default `ON`) — KServe clients + CLI
  plumbing. `OFF` → no KServe code, no Protobuf/gRPC needed.
- `NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS` (default `ON`) — local in-process
  engines; the **only** thing that fetches `neuriplo` (and the ONNX/TensorRT/
  LibTorch runtimes). `OFF` → no neuriplo fetch; the inference contract comes
  from the app-local headers in `app/inc/contract/` and `--kserve_endpoint`
  becomes mandatory at runtime.
- `NEURIPLO_INFER_ENABLE_GRPC` — gates only the gRPC transport.
- `NEURIPLO_INFER_ENABLE_KSERVE_TLS` (default `ON` when OpenSSL is found) —
  HTTPS for the HTTP client; without it `https://` endpoints fail fast with a
  clear "built without TLS support" error.

At least one of `ENABLE_KSERVE` / `ENABLE_LOCAL_BACKENDS` must be `ON`
(enforced at configure time):

| KSERVE | LOCAL_BACKENDS | Fetches neuriplo? | Result |
|:------:|:--------------:|:-----------------:|--------|
| ON  | ON  | yes | local engines + remote KServe (default) |
| OFF | ON  | yes | local engines only (no KServe code) |
| ON  | OFF | **no**  | KServe-only; contract from `app/inc/contract/` |
| OFF | OFF | — | configure error |

A KServe-only build still uses OpenCV and `neuriplo-tasks` (the image pipeline
and task layer); what it avoids is the `neuriplo` backend repo and its
per-backend runtimes. `neuriplo-tasks` does not depend on `neuriplo`.

## Constraints

- `InferenceMetadata` / `LayerInfo` come from the external `neuriplo` backend
  library and carry no datatype field, so datatypes are captured and held
  inside the KServe clients.
- `TensorElement` is `std::variant<float, int32_t, int64_t, uint8_t>`; output
  decoding is bounded to those four C++ types, with wider server datatypes
  widened/narrowed into them.

## Testing & CI

- Protocol helpers, retry policy, and security helpers are unit-tested in
  `neuriplo-kserve-client`; the adapter is covered by
  `app/test/test_KserveEngine.cpp` (fake client).
- `app/test/kserve_integration.sh` drives a KServe V2 round-trip against
  containerized Triton and OVMS over HTTP and gRPC. Dry-run mode runs on every
  PR (CTest: `kserve_integration_dry_run`; workflow:
  `.github/workflows/kserve-integration.yml`); the live path is gated behind a
  manual `workflow_dispatch` (`run_live=true`).
- The CI build matrix (`ci.yml` `kserve-build-matrix`) builds with and without
  gRPC, with `ENABLE_KSERVE=OFF`, and a kserve-only (`LOCAL_BACKENDS=OFF`)
  config, asserting KServe sources are absent in local-only builds and
  neuriplo is not fetched in kserve-only builds.

## Not implemented / out of scope

- Streaming inference.
- Server-side batching configuration.
- CLI exposure of the model-management API (the capability lives on the
  client; no admin subcommand is wired into the inference CLI).
