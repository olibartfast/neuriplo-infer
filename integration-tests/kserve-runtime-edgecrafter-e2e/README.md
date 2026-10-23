# EdgeCrafter KServe Runtime E2E

> Coverage owner: [docs/BackendCoverage.md](../../docs/BackendCoverage.md#integration-test-evidence) — this README documents the runner; coverage claims live there.

Validation status: Real local E2E (onnx_runtime, tensorrt, openvino, and executorch backends)

Version set: sibling branch set with local built binaries and model artifacts.

Owning repos:

- `neuriplo-infer`: owns this test and the app-layer KServe V2 client binary.
- `neuriplo-kserve-runtime`: KServe V2 serving runtime process.
- `neuriplo`: backend abstraction and ONNX Runtime / TensorRT / OpenVINO / ExecuTorch execution.
- `neuriplo-tasks`: EdgeCrafter (`ecdet`) preprocessing and postprocessing.

## Purpose

Exercise the EdgeCrafter detection dual-input contract through the real serving
path, across backends:

```text
images            FP32  [1,3,640,640]
orig_target_sizes INT64 [1,2]            -> labels INT64 [1,300]
                                            boxes  FP32  [1,300,4]
                                            scores FP32  [1,300]
```

This is the regression coverage for the metadata dtype-propagation fix: the test
asserts the advertised datatypes (`orig_target_sizes`/`labels` must be `INT64`,
not `FP32`). Before the fix the backend metadata hardcoded `FP32`, so the INT64
input was rejected by the runtime element-count check and the INT64 output
decoded as garbage. The ONNX Runtime, TensorRT, OpenVINO, and ExecuTorch backends are
validated.

## Model repository layout

Each backend has its own Triton-style model folder named
``<model>_<size>_<backend>``:

```text
${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_onnx/1/model.onnx
${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_tensorrt/1/model.engine
${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_openvino/1/model.{xml,bin}
${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_executorch/1/model.pte
```

Default repository root: `~/model_repository` (override with
`NEURIPLO_MODEL_REPOSITORY`).

Prepare or refresh all backend folders:

```bash
integration-tests/kserve-runtime-edgecrafter-e2e/prepare_model_repository.py
```

The prepare script copies ONNX and TensorRT engine sources when present, runs
`ovc` into `ecdet_s_openvino/1/` when IR is missing, and exports or copies
`model.pte` into `ecdet_s_executorch/1/` when missing. Pass `--backend` to
prepare a single backend folder.

## Required Local Artifacts

- runtime binaries (build per backend):
  - `../neuriplo-kserve-runtime/build/real-onnx/neuriplo-kserve-runtime`
  - `../neuriplo-kserve-runtime/build/real-trt/neuriplo-kserve-runtime`
  - `../neuriplo-kserve-runtime/build/real-openvino/neuriplo-kserve-runtime`
  - `../neuriplo-kserve-runtime/build/real-executorch/neuriplo-kserve-runtime`
  - gRPC variants: `real-onnx-grpc`, `real-trt-grpc`, `real-openvino-grpc`, `real-executorch-grpc`
- app binaries:
  - KServe client: `../neuriplo-infer/build-kserve-codex/app/neuriplo-infer`
  - local OpenVINO: `../neuriplo-infer/build-openvino/app/neuriplo-infer`
  - local ExecuTorch: `../neuriplo-infer/build-executorch/app/neuriplo-infer`
- model repository under `~/model_repository/ecdet_s_<backend>/1/` (see above)
- `../neuriplo-infer/data/dog.jpg` and `../neuriplo-infer/labels/coco.names`

Build the OpenVINO runtime from `neuriplo-kserve-runtime`:

```bash
cmake --preset real-openvino
cmake --build --preset real-openvino
cmake --preset real-openvino-grpc
cmake --build --preset real-openvino-grpc
```

Build the local OpenVINO app binary from `neuriplo-infer` (required for
`--mode local`):

```bash
cmake -S . -B build-openvino -G Ninja \
  -DDEFAULT_BACKEND=OPENVINO \
  -DNEURIPLO_INFER_ENABLE_KSERVE=ON
cmake --build build-openvino --parallel
```

When a sibling `../neuriplo` checkout exists, `neuriplo-infer` uses it automatically
so local OpenVINO and ExecuTorch pick up the same backend fixes as the runtime build.

Build the ExecuTorch runtime from `neuriplo-kserve-runtime` (requires
`~/dependencies/executorch` from `neuriplo/scripts/setup_executorch.sh`):

```bash
cmake --preset real-executorch
cmake --build --preset real-executorch
cmake --preset real-executorch-grpc
cmake --build --preset real-executorch-grpc
```

Build the local ExecuTorch app binary from `neuriplo-infer` (required for
`--mode local` with `--backend executorch`):

```bash
cmake -S . -B build-executorch -G Ninja \
  -DDEFAULT_BACKEND=EXECUTORCH \
  -DNEURIPLO_INFER_ENABLE_KSERVE=ON
cmake --build build-executorch --parallel
```

Export the ONNX model per `neuriplo-tasks/export/detection/edgecrafter/README.md`,
copy or convert artifacts with `prepare_model_repository.py`, and build the
TensorRT engine from the same ONNX with `trtexec` when exercising `tensorrt`.

Latency measurements (local in-process TRT vs KServe HTTP/gRPC, including the
HTTP binary-tensor fix) are recorded in [BENCHMARK.md](BENCHMARK.md).

LiteRT/TFLite is not part of the passing `ecdet` matrix. The LiteRT backend can
load the converted graph with custom `ONNX_GRIDSAMPLE` and integer `SIGN`
kernels, but the ONNX-to-TFLite conversion currently produces invalid detection
scores and boxes for this deformable/transformer model. Keep `ecdet` on ONNX
Runtime, TensorRT, or OpenVINO until conversion fidelity is fixed.

## Usage

From `neuriplo-infer`:

```bash
# Prepare model repository (once)
integration-tests/kserve-runtime-edgecrafter-e2e/prepare_model_repository.py

# ONNX Runtime backend (HTTP transport)
integration-tests/kserve-runtime-edgecrafter-e2e/run.py --backend onnx_runtime

# TensorRT backend (HTTP transport)
integration-tests/kserve-runtime-edgecrafter-e2e/run.py --backend tensorrt

# OpenVINO backend over HTTP
integration-tests/kserve-runtime-edgecrafter-e2e/run.py --backend openvino

# OpenVINO in-process (no KServe hop)
integration-tests/kserve-runtime-edgecrafter-e2e/run.py --backend openvino --mode local

# OpenVINO local + HTTP + gRPC in one invocation
integration-tests/kserve-runtime-edgecrafter-e2e/run_openvino_matrix.py

# ExecuTorch backend over HTTP
integration-tests/kserve-runtime-edgecrafter-e2e/run.py --backend executorch

# ExecuTorch in-process (no KServe hop)
integration-tests/kserve-runtime-edgecrafter-e2e/run.py --backend executorch --mode local

# ExecuTorch local + HTTP + gRPC in one invocation
integration-tests/kserve-runtime-edgecrafter-e2e/run_executorch_matrix.py

# TensorRT backend over gRPC (needs a runtime built with gRPC support)
integration-tests/kserve-runtime-edgecrafter-e2e/run.py --backend tensorrt --transport grpc
```

Select the client transport with `--transport {http,grpc}` (default `http`) in
`--mode kserve`. The `grpc` transport requires the runtime binary to be built
with `-DNEURIPLO_RUNTIME_ENABLE_GRPC=ON`; the runtime then serves gRPC on
`--grpc-port` (default `--port + 1`) alongside HTTP. The HTTP E2E path enables
the KServe binary tensor extension for inference requests so large image tensors
travel as raw bytes instead of JSON number arrays; gRPC still uses protobuf raw
tensor contents by default.

The `tensorrt` backend adds the TensorRT library directory to `LD_LIBRARY_PATH`
for the runtime process; override the default with `--tensorrt-lib-dir` (it is
repeatable). The `openvino` backend adds
`$OPENVINO_DIR/runtime/lib/intel64` (default `~/dependencies/openvino_2025.2.0`).
The `executorch` backend adds `$EXECUTORCH_DIR/lib` (default
`~/dependencies/executorch`). Override model and binary paths with `--model`, `--runtime-bin`, `--infer-bin`,
and `--local-infer-bin` when using non-default locations.

## What It Checks

- runtime starts with the requested backend; `/v2/health/{live,ready}` are true
- `/v2/models/ecdet_s_<backend>` reports `neuriplo_<backend>`, inputs `images` +
  `orig_target_sizes`, outputs `labels` / `boxes` / `scores`
- advertised datatypes match the contract (`orig_target_sizes` and `labels` are
  `INT64`; `images`, `boxes`, `scores` are `FP32`) - the dtype regression guard
- the app-layer executable runs a real KServe inference request over the
  selected transport (HTTP or gRPC), or in-process OpenVINO/ExecuTorch for `--mode local`
- `../neuriplo-infer/data/output/processed_ecdet_kserve.png` (KServe) or
  `processed_ecdet_openvino.png` / `processed_ecdet_executorch.png` (local) is refreshed and non-empty
- Prometheus metrics confirm one successful request and zero failures: the HTTP
  per-transport counters for `--transport http`, and the transport-agnostic
  `neuriplo_scheduler_requests_{accepted,rejected,timed_out}_total` for
  `--transport grpc` (the gRPC server has no per-transport counter)

## Notes

- The rendered image is named `processed_<model>_<backend>.png` by the app,
  where backend is always `kserve` for remote inference (the client does not
  observe the runtime's compute backend), so KServe cases write
  `processed_ecdet_kserve.png`. Local OpenVINO writes `processed_ecdet_openvino.png`;
  local ExecuTorch writes `processed_ecdet_executorch.png`.
- TensorRT engines are TensorRT-version and GPU specific; a runtime built
  against a different TensorRT version will fail to deserialize the engine.
- OpenVINO IR is served from `ecdet_s_openvino/1/model.xml` (or `model.onnx` in
  that folder when IR conversion is skipped).
- ExecuTorch serves `ecdet_s_executorch/1/model.pte` exported from the EdgeCrafter PyTorch checkpoint via
  `export_ecdet_executorch.py` (or copied by `prepare_model_repository.py`). Requires
  `edgecrafter-cpp-inference/third_party/EdgeCrafter` and `ecdet_s.pth`.
