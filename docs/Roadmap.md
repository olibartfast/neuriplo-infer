# Roadmap

## Current State (v0.3.0-dev)

The vision-stack cluster consists of four repos with clean separation of concerns:

| Repo | Role |
|---|---|
| **vision-core** | Task contracts, pre/postprocessing, result types |
| **neuriplo** | Backend orchestration (OPENCV_DNN, ONNX_RUNTIME, LIBTORCH, TENSORRT, OPENVINO, LIBTENSORFLOW, GGML, TVM) |
| **videocapture** | Video I/O (OpenCV, GStreamer, FFmpeg) |
| **vision-inference** | Application layer, CLI, visualization |

A sibling application repo, [vision-tracking](https://github.com/olibartfast/vision-tracking), handles detection + tracking pipelines using the same shared libraries. It maintains its own ops control plane independently.

## Phase 1: Solidify Foundations

### Testing & Reliability
- Expand CI test coverage beyond OPENCV_DNN — add at minimum ONNX_RUNTIME integration tests
- Add output-consistency tests validating result schema stability across backends
- Add cross-backend numerical-drift checks within defined tolerance

### Automation Wiring
- Wire `policies.yaml` and `CLUSTER_MAP.yaml` into CI to validate PRs against allowed/forbidden change classes
- Enforce `PR_EVIDENCE_TEMPLATE.md` on agent-generated PRs
- Add cross-repo version-drift check: validate all repos' `develop` branches are compatible on every push

### Performance Baseline
- Implement the benchmark smoke command as a real CI gate
- Store latency/throughput baselines per backend to catch regressions (5% threshold defined in policies)

## Phase 2: Scale Backend & Model Coverage

### Backend Maturity
- Promote GGML and TVM backends in neuriplo from experimental to tested
- Add CUDA Execution Provider testing for ONNX Runtime in CI
- Evaluate CoreML or QNN backend for mobile/edge inference

### Model Coverage — 2D
- Expand VLM/multi-modal support beyond OWLv2 (tokenizer infrastructure already in CLI)
- Add OCR/text detection task types (in vision-core)

### Model Coverage — 3D
- **Monocular 3D detection** (MonoDETR, FCOS3D) — single image input, new `Detection3D` result variant in vision-core. Easiest entry point for 3D; no changes needed in neuriplo or videocapture
- **Stereo depth** (RAFT-Stereo, CREStereo) — multi-frame input already supported via RAFT optical flow pattern. Reuses existing `DepthEstimation` result type
- **BEV detection** (BEVFormer, PointPillars) — evaluate once monocular 3D is stable. May require new input abstractions in vision-core for point cloud data

> Note: NeRF/Gaussian Splatting and real-time SLAM are iterative/stateful pipelines that don't fit the single-pass `TaskInterface` model. These belong in a separate repo if pursued.

### Video Pipeline
- Add RTSP/WebRTC source support in videocapture for real-time deployment
- Optimize batched video inference per-backend

## Phase 3: Production & Deployment

### Serving Layer
- Add lightweight HTTP/gRPC serving mode (new repo or optional build target)
- Support model hot-swapping and multi-model routing

### Containerization & Orchestration
- Health checks, Prometheus metrics, and structured logging
- Kubernetes/Docker Compose deployment manifests
- Model registry integration (MLflow, Triton model repo format)

### Agentic Operations
- Evolve ops control plane from documentation to executable CI workflows
- Automate CI triage and cross-repo migration runbooks as agent workflows
- Automated release orchestration across the cluster: version bumps, changelog generation, compatibility matrix

### Observability
- Inference telemetry: latency histograms, throughput counters, error rates per backend
- Model accuracy drift detection for deployed models

## Priority Matrix

| Initiative | Impact | Effort | Priority |
|---|---|---|---|
| Cross-backend CI tests | High | Medium | P0 |
| Performance baseline in CI | High | Low | P0 |
| Policy enforcement in CI | Medium | Low | P1 |
| GGML/TVM backend promotion | High | High | P1 |
| Monocular 3D detection | High | Medium | P1 |
| Stereo depth models | Medium | Medium | P2 |
| Serving layer (HTTP/gRPC) | High | High | P2 |
| VLM/multi-modal expansion | Medium | Medium | P2 |
| Agent workflow automation | Medium | High | P3 |
| Kubernetes deployment | Medium | Medium | P3 |
| BEV / point cloud models | Medium | High | P3 |
