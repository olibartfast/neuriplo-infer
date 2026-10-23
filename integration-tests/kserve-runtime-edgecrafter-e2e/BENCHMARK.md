# EdgeCrafter (`ecdet`) KServe latency benchmark

Recorded measurements from the Jun 12-13, 2026 local validation work on
`ecdet_s` (EdgeCrafter detection, small variant) through the Neuriplo KServe
stack. This is release evidence for transport choice on large image tensors, not
a CI regression gate.

## Setup

| Item | Value |
|------|-------|
| Model | `~/model_repository/ecdet_s_tensorrt/1/model.engine` (EdgeCrafter v1 small; built with TensorRT 10.13.3.9) |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| CPU | 11th Gen Intel Core i5-11400H |
| Input | `neuriplo-infer/data/dog.jpg` (576x768, resized to 640x640) |
| Labels | `neuriplo-infer/labels/coco.names` |
| Runtime | `neuriplo-kserve-runtime` with `--backend tensorrt` |
| Client | `neuriplo-infer` KServe mode (`--kserve_transport={http,grpc}`) |

The `images` input is `1x3x640x640` FP32 (~1.23 M floats, ~4.7 MB raw). That
size dominates transport cost when tensors are serialized as text.

## Results (client round-trip)

Same engine and GPU for all rows. Server-side compute was isolated with
`neuriplo_scheduler_infer_seconds` (~101-111 ms), matching local in-process
TRT (~90 ms).

| Path | Client round-trip | Notes |
|------|------------------:|-------|
| **Local in-process TRT** | **~90 ms** (87 / 90 / 91) | `neuriplo-infer` loads the engine directly; no KServe hop |
| **Server - HTTP/JSON** (default client) | **~1375 ms** (1461 / 1356 / 1375) | `KSERVE_BINARY` unset; JSON float array for `images` |
| **Server - gRPC** | **~84-94 ms** warm (136 ms first call) | Raw protobuf tensor bytes; channel setup on first call |
| **Server - HTTP + binary extension** | **~142 ms** (e2e verification) | `KSERVE_BINARY=1`; see fix below |

Earlier broken runs (before dtype propagation was fixed) showed even higher HTTP
latencies (e.g. 1536 ms, 6363 ms) because INT64 tensors were mis-advertised as
FP32; those numbers are not comparable to the table above.

## Root cause: HTTP JSON encoding

The default KServe HTTP client path in `neuriplo-kserve-client` serializes tensor
payloads as JSON number arrays (`encodeTensorData` -> `request.dump()`). For
`images`, that inflates ~4.7 MB binary into tens of MB of text that must be
built, sent, and parsed on both ends. Runtime compute stayed ~100 ms; the
~1.27 s gap was almost entirely transport.

gRPC sends raw bytes in protobuf by default, which is why server gRPC matched
local in-process latency.

## Fix: HTTP binary tensor extension (Codex session)

The client and runtime already supported the KServe **binary tensor data
extension**; it was opt-in on HTTP via `KSERVE_BINARY=1` (see
`neuriplo-kserve-client/src/KserveHttpClient.cpp`).

The EdgeCrafter E2E runner was updated in commit `4f8314d` (`test(ecdet): use
HTTP binary tensors`) to set `KSERVE_BINARY=1` whenever `--transport http`:

```python
if args.transport == "http":
    infer_env["KSERVE_BINARY"] = "1"
```

That change closes the HTTP/JSON overhead for this workload. After the fix,
`(tensorrt, http)` and `(tensorrt, grpc)` both pass the E2E runner; gRPC
verification logged **142 ms** client round-trip on the same hardware.

For manual runs outside the E2E script:

```bash
export KSERVE_BINARY=1
./neuriplo-infer --type=ecdet --source=data/dog.jpg --labels=labels/coco.names \
  --kserve_endpoint=http://127.0.0.1:8080 --kserve_model_name=ecdet_s_tensorrt \
  --kserve_transport=http
```

gRPC does not need `KSERVE_BINARY` (raw contents are the default).

Related runtime release notes: `neuriplo-kserve-runtime` v0.2.0 added HTTP
binary tensor framing on the server side; see that repo's CHANGELOG.

## How to reproduce

From `neuriplo-infer` (requires local binaries and `ecdet_s_tensorrt/1/model.engine`):

```bash
# Correctness + transport smoke (HTTP uses KSERVE_BINARY=1 in the runner)
integration-tests/kserve-runtime-edgecrafter-e2e/run.py --backend tensorrt
integration-tests/kserve-runtime-edgecrafter-e2e/run.py --backend tensorrt --transport grpc

# Latency comparison (manual; run multiple times for stability)
# 1. Local in-process TRT (configure neuriplo-infer with DEFAULT_BACKEND=TENSORRT)
# 2. Server HTTP without binary (omit KSERVE_BINARY) - expect JSON overhead
# 3. Server HTTP with KSERVE_BINARY=1 - expect near-local latency
# 4. Server gRPC - expect near-local latency
```

Use `--benchmark --iterations=N` on `neuriplo-infer` for repeated local
in-process samples. For server paths, read `Inference time:` from client logs and
`neuriplo_scheduler_infer_seconds` from runtime `/metrics`.

## Provenance

| Session | What it captured |
|---------|------------------|
| Claude Code `24a185f3-e720-4395-8c00-2059514fd95c` (2026-06-12) | Local vs server TRT; HTTP/JSON vs gRPC decomposition; dtype fix and E2E runner with `--transport grpc` |
| Codex / platform commit `4f8314d` (2026-06-13) | E2E runner enables `KSERVE_BINARY=1` for HTTP; documents binary path in README |

Primary session log (local only):

`~/.claude/projects/-home-oli-repos-neuriplo-platform/24a185f3-e720-4395-8c00-2059514fd95c.jsonl`

## Caveats

- Numbers are from one machine and one image; treat as directional, not SLOs.
- TensorRT engines are GPU- and TensorRT-version-specific.
- First gRPC call includes channel setup (~50 ms extra).
- Platform policy thresholds (`ops/policies.yaml`: 5% latency regression) apply
  to owned-repo benchmarks; this doc is evidence, not an automated gate.
