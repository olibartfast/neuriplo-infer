# Feature Requirements - TensorFlow GPU and remote TPU

**Date:** 2026-11-09
**Branch:** `feature/tensorflow-gpu-remote-tpu` from `develop`
**Status:** Implemented; live GPU/TPU evidence pending
**Cross-repo packet:** `2026-11-09-tensorflow-gpu-remote-tpu` in `neuriplo`,
`neuriplo-tasks`, and `neuriplo-platform`

Roadmap phase: accelerator coverage slice; add the packet to
[`../roadmap.md`](../roadmap.md) when the branch merges.

## Goal

Expose two accelerator paths through the application without adding a new
device-selection contract:

1. A local NVIDIA GPU path for the TensorFlow backend, using the existing
   `--use-gpu` flag and a GPU-capable `neuriplo-infer:tensorflow-gpu` image.
2. A remote TPU path that uses the existing KServe V2 client
   (`--kserve_endpoint`, `--kserve_model_name`, `--kserve_transport`) against a
   TPU-side serving process.

The application must keep its current CPU default, output schema, task routing,
and `--capabilities` schema.

## In Scope

- `docker_run_inference_e2e_example.sh` support for `tensorflow` and
  `tensorflow-gpu` backends, including a `tensorflow_classifier` preset.
- A SavedModel directory mount path and an export step that calls the existing
  `neuriplo-tasks` TensorFlow classifier exporter.
- `--gpus=all` plus `--use-gpu` only for the `tensorflow-gpu` backend; the CPU
  `tensorflow` backend remains the default for that preset.
- An `integration-tests/tpu-kserve/` harness with `--dry-run` and `--live` modes,
  baseline/evidence files, and a CTest dry-run entry.
- GPU and TPU documentation: `docs/Deployment.md`, `docs/Usage.md`,
  `docs/BackendCoverage.md`, `docs/KserveCompatibility.md`, and `README.md`
  links where the existing docs already point.
- Focused dry-run tests and a manual live-evidence path.

## Out of Scope

- An in-process TPU/PJRT backend. TPU is remote serving only.
- A new `--device` flag or a change from `use_gpu` to a device enum.
- Shape, dtype, routing, preprocessing, postprocessing, or output-schema changes.
- Changes to `docs/capabilities.schema.json` or `--capabilities` output.
- Changes to `neuriplo-tasks` task code, `videocapture`, `neuriplo-track`, or
  `neuriplo-ui`.
- Source changes in `neuriplo-kserve-client` or `neuriplo-kserve-runtime`; the
  existing KServe V2 path is sufficient for the first TPU representative.
- Multi-GPU selection, TPU pod/multi-host serving, or training.
- A production TPU serving runtime. The TPU service is operator-supplied or a
  reference process documented by the runbook; this packet owns only the client
  and evidence path.
- Claiming a PASS before the manual GPU or TPU run is recorded.

## Decisions

- [D-1] Keep `--use-gpu` as the only local accelerator flag. The TensorFlow GPU
  path is a backend/image choice, not a new CLI contract.
- [D-2] Name the e2e backend `tensorflow-gpu` and the image
  `neuriplo-infer:tensorflow-gpu`. This mirrors the existing
  `neuriplo-infer:<backend>` convention and avoids implicit GPU selection.
- [D-3] Use the existing KServe V2 client for TPU. The server owns device
  placement; the client selects a model name and endpoint, not a TPU device.
- [D-4] Keep TPU live checks out of routine CI. Hosted CI runs only the
  `--dry-run` harness and the existing KServe dry-run path.
- [D-5] Add a separate accelerator evidence section/table in
  `docs/BackendCoverage.md` instead of changing CPU cells without evidence.
- [D-6] The TensorFlow e2e preset is classifier-only. Detection and segmentation
  TensorFlow presets are follow-ups.

## Requirements

- [R-1] `docker_run_inference_e2e_example.sh` accepts `tensorflow` and
  `tensorflow-gpu` backends and mounts a SavedModel directory as `--weights`.
- [R-2] The `tensorflow_classifier` preset runs on CPU with the `tensorflow`
  backend and on GPU with the `tensorflow-gpu` backend; the CPU path remains the
  default.
- [R-3] Only the `tensorflow-gpu` backend adds `--gpus=all` to Docker and
  `--use-gpu` to the CLI.
- [R-4] `integration-tests/tpu-kserve/` provides a dry-run command that needs no
  network or hardware, and a live mode that drives `neuriplo-infer` against a
  KServe V2 TPU endpoint.
- [R-5] The TPU live check remains outside routine CI and is recorded as
  attested evidence, not as an inferred pass.
- [R-6] `docs/Deployment.md`, `docs/Usage.md`, `docs/KserveCompatibility.md`, and
  `docs/BackendCoverage.md` describe the paths, prerequisites, and limits without
  claiming unverified results.
- [R-7] Task routing, preprocessing, postprocessing, output schema, and
  `--capabilities` output remain unchanged.
- [R-8] The existing app test suite and e2e dry-run tests stay green without a
  GPU or TPU.

## Constraints and Context

- The current e2e script supports only `onnxruntime`, `tensorrt`, `litert`,
  `executorch`, and `llamacpp` backends. TensorFlow support is new plumbing, not
  a one-line preset addition.
- The current TensorFlow app image is CPU-only (`docker/Dockerfile.libtensorflow`
  in this repo; the `neuriplo` test image is separate). The e2e script selects
  the image by backend; the GPU variant is an opt-in build arg, not a new
  contract.
- The current KServe client path already supports raw FP32 tensors and HTTP/gRPC.
  The TPU service must expose KServe V2; no client protocol change is planned.
- The existing `rfdetr_pose_evidence_scaffold` pattern is the model for a
  dry-run-only evidence scaffold registered with CTest.
- `README.md` links to `docs/Deployment.md` and `docs/BackendCoverage.md`; keep
  those links valid and avoid duplicating the full matrix in the README.

## Open Questions

- [Q-1] Should `tensorflow_classifier` use `data/dog.jpg` and
  `labels/imagenet_labels.txt` as the representative? Recommendation: yes,
  matching the E6 classifier evidence.
- [Q-2] Does the TPU service expose HTTP only or HTTP plus gRPC? The harness
  should cover HTTP first; add gRPC only if the service implements it.
- [Q-3] Does the GPU image require a separate compose file, or is the existing
  `docker run --gpus all` invocation sufficient? Recommendation: the existing
  invocation plus a documented build command; no compose file.
