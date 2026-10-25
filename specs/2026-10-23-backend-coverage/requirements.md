# Feature Requirements — Backend coverage matrix

Roadmap phase: [Phase 5 — Make Triton/OVMS compatibility evidence routine](../roadmap.md#phase-5--make-tritonovms-compatibility-evidence-routine) (related; this packet does not close it)
Branch: `feature/backend-coverage`

## Goal

Publish a `docs/BackendCoverage.md` matrix styled on the NERT
`specs/2026-10-17-task-models/coverage.md` that records, for every task family
in `docs/generated/supported-model-types.md`, its status on every supported
backend. A reader can tell what runs where, what the evidence is, and what is
still untested — without reading shell, workflows, and three docs.

## In Scope

- `docs/BackendCoverage.md` with: Definitions, backend capability table
  (artifact per backend), task × backend gate table (10 task rows × 10 columns:
  `OPENCV_DNN`, `ONNX_RUNTIME`, `TENSORRT`, `LIBTORCH`, `OPENVINO`,
  `LIBTENSORFLOW`, `LITERT`, `EXECUTORCH`, `LLAMACPP`, `KSERVE`),
  release-style scoreboard, open items, non-kernel limits, sources.
- Representative model per task row taken from `docker_run_inference_e2e_example.sh`
  presets (`rtdetrv4`, `owlv2`, `torchvision_classifier`, `yoloseg`,
  `raft`, `vitpose`/`rfdetr_keypoint`, `depth_anything_v2`, `videomae`,
  `gemma4`, `yolo26s_tflite`/`yolov8_executorch`, `ecdet`/`ecseg`/`ecpose`).
- Cell semantics NERT-like: `PASS` = live end-to-end run with recorded expected
  output; `DRY-RUN` = preset exists and `e2e-example-emulation.yml` is green;
  `BUILDS` = CI `ci.yml` compiles the backend; `KNOWN-BAD` = runs but numerically
  wrong with pointer to `docs/KserveCompatibility.md`; `UNTESTED` otherwise.
  No cell claims more than its evidence.
- Links from `README.md` docs table and `docs/Deployment.md` backends section.
- `CHANGELOG.md` entry under `[Unreleased]`.

## Out of Scope

- Live runs for every cell. This branch records existing evidence only; filling
  `UNTESTED` cells is follow-up work (tracked in the doc's Open items).
- New export tooling, new presets, new CI live jobs, new Dockerfiles.
- `--capabilities` output, `docs/capabilities.schema.json`, generated
  `docs/generated/supported-model-types.md` — untouched.
- Backend SDK versions (owned by `neuriplo` per `tech-stack.md`).
- Closing roadmap Phase 5 (cadence for live Triton/OVMS runs stays open).

## Decisions

- Columns are the 9 local backends from `docs/Deployment.md` plus `KSERVE`
  (remote `KserveEngine`), because `README.md` Key Features advertises remote
  KServe mode as a backend choice and `KserveCompatibility.md` already tracks it.
  Sibling-only backends (NERT, MIGRAPHX, DALI, CACTUS, GGML, TVM) are out:
  this repo cannot build them.
- Rows are task families, not every `--type` alias. The alias inventory stays
  generated; the matrix maps one gated representative per family and names the
  preset that gates it. Same convention as NERT's scoreboard note (one
  representative closes the entry).
- `PASS` requires a recorded live output (detections, labels, or round-trip),
  not a green build or dry-run. This keeps the matrix honest: most cells land
  on `BUILDS` / `DRY-RUN` / `UNTESTED` today.
- Manual doc, not generated. The `sync_supported_model_types.py --check` gate
  covers only the generated alias list; a generated matrix would need a new
  generator plus CI wiring — deferred, noted in Open items.
- `rfdetr_keypoint` preset exists in the script but is missing from the
  `e2e-example-emulation.yml` matrix. Recorded as a gap in Open items, not
  fixed here.

## Constraints and Context

- `specs/tech-stack.md`: no backend package versions in prose; no hand-editing
  `docs/generated/supported-model-types.md`; no tensor-shape/result-schema
  semantics (pointer to `neuriplo-tasks` only); docs hyperlinks must resolve
  (`ls` relative, `curl -sI` absolute).
- Reuse: `docs/Deployment.md` backends table for artifact formats,
  `docs/KserveCompatibility.md` for KServe/server cells, NERT `coverage.md`
  Definitions / Scoreboard / Open-items structure.
- Cross-repo impact: none. No `neuriplo` / `neuriplo-tasks` / `videocapture` /
  `neuriplo-kserve-client` change or pin move.
- Contract surfaces touched: `README.md` docs table, `docs/Deployment.md`
  backends section, `CHANGELOG.md`. No CLI, no `--capabilities`, no schema.

## Open Questions

- None blocking. Follow-ups live in `docs/BackendCoverage.md` Open items:
  live-gate each `DRY-RUN` cell, add the missing `rfdetr_keypoint` emulation
  job, decide a live-run cadence (Phase 5).
