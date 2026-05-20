# Roadmap

## Current State (v0.3.0-dev)

The vision-stack cluster consists of four repos with clean separation of concerns:

| Repo | Role |
|---|---|
| **vision-core** | Task contracts, pre/postprocessing, result types |
| **neuriplo** | Backend orchestration for the backends currently wired into this app, plus additional backend experiments owned upstream |
| **videocapture** | Video I/O (OpenCV, GStreamer, FFmpeg) |
| **vision-inference** | Application layer, CLI, visualization |

Sibling application repos consume vision-core independently:

| Sibling Repo | Role | vision-core consumer? |
|---|---|---|
| [vision-tracking](https://github.com/olibartfast/vision-tracking) | Detection + tracking pipelines | Yes |
| [tritonic](https://github.com/olibartfast/tritonic) | Triton Inference Server client for CV tasks | Yes |

Both maintain their own ops control planes independently — vision-inference does not depend on them.

For canonical current repo boundaries and public surfaces, prefer `ops/` metadata over this roadmap.

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
- Add a multimodal understanding path for Gemma 4-class models with freeform text plus optional structured outputs
- Prefer a Cactus-backed on-device path in `neuriplo` for Gemma 4 image understanding, with `llama.cpp` retained as a fallback runtime candidate
- Add OCR/text detection task types (in vision-core)

### Model Coverage — 3D
- **Monocular 3D detection** (MonoDETR, FCOS3D) — single image input, new `Detection3D` result variant in vision-core. Easiest entry point for 3D; no changes needed in neuriplo or videocapture
- **Stereo depth** (RAFT-Stereo, CREStereo) — multi-frame input already supported via RAFT optical flow pattern. Reuses existing `DepthEstimation` result type
- **BEV detection** (BEVFormer, PointPillars) — evaluate once monocular 3D is stable. May require new input abstractions in vision-core for point cloud data

> Note: NeRF/Gaussian Splatting and real-time SLAM are iterative/stateful pipelines that don't fit the single-pass `TaskInterface` model. These belong in a separate repo if pursued.

### Video Pipeline
- Add RTSP/WebRTC source support in videocapture for real-time deployment
- Optimize batched video inference per-backend
- Add multimodal video-understanding support using `videocapture` for frame decode/sampling and Gemma 4-style backends for captioning, QA, and event summarization
- Start with short local clips, sampled frames, and text-first outputs before adding audio or streaming support

### Multimodal Understanding
- Introduce new `vision-core` task/result contracts for `ImageUnderstanding` and `VideoUnderstanding` instead of overloading classification semantics
- Standardize outputs around a stable schema with required freeform `text` plus optional `answer`, grounded regions, and temporal events
- Keep V1 constrained to local files, uniform frame sampling, and parseable JSON/text outputs suitable for E2E regression tests
- Validate in this order: standalone runtime spike, upstream contract work in `vision-core`, backend integration in `neuriplo`, then CLI and E2E wiring in `vision-inference`
- Prioritize Cactus for mobile/on-device Gemma 4 support, especially for image understanding and short video understanding, with `llama.cpp` as the generic fallback path
- Design multimodal contracts in `vision-core` so they are consumable by all downstream apps (vision-inference, tritonic, vision-tracking) without backend-specific coupling
- Coordinate with tritonic's planned multimodal task mode: tritonic will consume the same `vision-core` multimodal contracts via Triton Server backends, so contract design must remain backend-agnostic and avoid assumptions about local model loading

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
| Gemma 4 multimodal spike via Cactus | High | Medium | P1 |
| Image/Video understanding contracts | High | Medium | P1 |
| Tritonic multimodal contract alignment | Medium | Low | P1 |
| GGML/TVM backend promotion | High | High | P1 |
| Monocular 3D detection | High | Medium | P1 |
| Stereo depth models | Medium | Medium | P2 |
| Video understanding with sampled frames | High | Medium | P2 |
| Serving layer (HTTP/gRPC) | High | High | P2 |
| VLM/multi-modal expansion | High | Medium | P2 |
| Agent workflow automation | Medium | High | P3 |
| Kubernetes deployment | Medium | Medium | P3 |
| BEV / point cloud models | Medium | High | P3 |
