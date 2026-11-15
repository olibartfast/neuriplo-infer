# Feature Requirements — GGML matrix coverage

Roadmap phase: [Phase 5 — Make Triton/OVMS compatibility evidence routine](../roadmap.md#phase-5--make-tritonovms-compatibility-evidence-routine) (related; this packet does not close it) + `docs/BackendCoverage.md` Open items follow-up
Branch: `feature/ggml-matrix-coverage`

## Goal

Give the GGML backend the same honest matrix treatment the MGX column already has: a `GGML` column in `docs/BackendCoverage.md` that claims no more than its evidence. Attempt a full live gate on a brev CPU instance (`-DDEFAULT_BACKEND=GGML` configure, build, per-task runs); record whatever the run actually proves. The expected honest outcome is `UNTESTED`/`BLOCKED` cells plus a dedicated evidence section — not `PASS` — because the upstream implementation is a stub.

## In Scope

- R-1 — honest GGML column (MGX pattern): `GGML` legend entry disambiguated
  from `LLAMACPP`; `GGML` column in the second gate table with every task
  cell `U` (no by-construction `N/A`); `no-ci-job` note; preamble + `Scope
  and caveats` paragraphs (upstream builder + engine-test green, no
  app-surface promotion); new `E19 — GGML coverage status` (Slurm jobs 29/30
  + `77/77` condensed out of `E11`; `E11` keeps LLaMA PASS + cross-ref);
  `Environments` GGML row; `Open items` GGML follow-up; `Last updated`
  bump; Scoreboard untouched.
- R-2 — brev live attempt with recorded evidence: on a cheap CPU instance
  (`cpu-e2.4vcpu-16gb`, x86_64, CPU-only matches `GGMLInfer::setup_backend`
  which forces CPU even with `use_gpu=true`):
  - `cmake -S . -B build-ggml -DDEFAULT_BACKEND=GGML` configure attempt against the pinned stack (`neuriplo v0.11.3` per `versions.env`; GGML SDK version owned by `neuriplo`, not repeated here).
  - Minimal app-side wiring probe if configure fails (report the exact missing piece; keep any edit to the smallest `LinkBackend.cmake` / `app/test/CMakeLists.txt` / `DEFAULT_BACKEND` help-string change that unblocks configure).
  - One representative live attempt (classification-shaped `3x224x224 F32 → 1000`, the only shape expressible) to record actual output (expected: zeros/placeholder, proving `BLOCKED`, not `PASS`).
  - Brev lifecycle: run the attempts, **delete the instance when testing completes** (validation V-5).
- R-3 — boundaries kept: no `versions.env` pin move, no CLI / `--capabilities`
  / schema / generated-docs change, no `neuriplo` inference-semantics change.
- R-4 — `CHANGELOG.md` entry under `[Unreleased]` (docs-only column, no `PASS` claimed).

## Out of Scope

- Fixing `neuriplo` `GGMLInfer` to load real GGML/GGUF models or to produce real outputs. Inference semantics live in `neuriplo` per `tech-stack.md`; changing them here is a forbidden change class — escalate to a human / raise in `neuriplo` instead. This packet proves the boundary, it does not cross it.
- `neuriplo` / `neuriplo-tasks` / `videocapture` / `neuriplo-kserve-client` pin moves (`versions.env` untouched).
- New `docker/Dockerfile.ggml` in this repo, new `ci.yml` `build-ggml` job, new e2e presets, `--capabilities` / schema / generated-docs changes, CLI contract changes.
- GPU testing for GGML (`setup_backend` forces CPU; `Dockerfile.ggml` builds `GGML_CUDA=OFF`, `GGML_METAL=OFF`).

## Decisions

- Full-live-gating was requested, so we attempt it on brev rather than asserting impossibility from source alone. The attempt is the evidence that justifies `BLOCKED`/`UNTESTED` instead of a guess.
- MGX is the structural template (contract-only preamble + all-`U` column + standalone `E`-section + `Environments` row + Open item), because it is the repo's accepted pattern for "backend exists upstream, no app-surface gate".
- `gemma4` cell value (`U` vs `N/A`) is decided in `plan.md` against the constructional rule (`N/A` = backend cannot serve this task by construction). GGML parses no model bytes (`GGMLInfer.cpp:77-89`), so no constructional exclusion applies the way MGX's ONNX-only scope does — default is `U` unless the brev run proves a loader-level `BLOCKED`.
- CPU-only brev instance: GGML has no GPU path, so a GPU instance would waste cost and prove nothing extra.

## Constraints and Context

- `specs/tech-stack.md`: no tensor-shape/dtype/result-schema semantics in this repo (pointer to `neuriplo-tasks` only); no backend package versions in prose (point at `versions.env` / `neuriplo`); no hand-editing generated docs; docs hyperlinks must resolve.
- `specs/architecture.md`: app owns CLI/config/wiring/visualization/flow; task contracts live in `neuriplo-tasks`, backend behavior in `neuriplo`. A GGML task-cell promotion would need both siblings to move first.
- `2026-10-23-backend-coverage/requirements.md:48-49` scoped GGML out ("sibling-only … this repo cannot build them"). This packet explicitly expands that scope to a docs column + build attempt; it does not promise a buildable backend.
- Cross-repo impact: none. No sibling change or pin move. If the brev attempt shows the stub cannot be gated, the follow-up belongs to `neuriplo` (real model loading), not here.
- Contract surfaces touched: `docs/BackendCoverage.md`, `CHANGELOG.md`. No CLI, no `--capabilities`, no schema, no generated model-type docs.

## Open Questions

- None blocking. The one empirical question (does `-DDEFAULT_BACKEND=GGML` configure/build on the pinned stack, and what does the single-shape run emit?) is answered by the brev attempt in `validation.md`, not by further discussion.

---

## Phase 2 — TEST-ALL-THE-MODELS (program, started 2026-11-13)

Maintainer scope: every representative model in the gate tables gets a real
GGML run. A cell flips to `PASS` only on a recorded run with ORT parity
(top-5/logit agreement per the neuriplo-side bar); anything else stays `U`
or becomes `BLOCKED` with the owning repo and reason named. No aspiration
cells. The `neuriplo` loader merge (PR #74, full ResNet18 + parity) is the
foundation; per-model graph work continues there, gating happens here.

### Phase 2 requirements (append-only: R-5+)

- [R-5] Each gate-table representative gets a GGML attempt on a CPU brev box
  (`-DDEFAULT_BACKEND=GGML` app build against a sibling containing the loader;
  ORT reference where the model has one): record PASS (with parity numbers)
  or BLOCKED (with owner + reason). `gemma4` stays out (VLM belongs to the
  `LLAMACPP` column; no GGML VLM claim).
- [R-6] Pin honesty: `versions.env` still pins `neuriplo v0.11.3`, which lacks
  the loader — so Phase 2 runs locally against the sibling `develop` checkout
  and says so in E19. No CI `build-ggml` job and no cell promotion on CI until
  a neuriplo release ships the loader and the pin moves (release cut is the
  maintainer's call, separate packet).
- [R-7] `E19` grows per-model evidence subsections (commands, artifacts, parity
  numbers, brev instance IDs); Scoreboard and `Last updated` move only on real
  PASS cells. Wiring stays as-is (no new flags, presets, or schema changes
  unless a model proves one necessary).
- [R-8] Order by feasibility; like-family models share graph builders.
  Inventory (2026-11-13, verified against export guides + E-sections):
  1. `torchvisionclassifier` ResNet-50 — same 7-op set as the merged ResNet18
     gate (only bottleneck 1x1s + 2048-d FC added); matches E6 exactly.
  2. YOLO conv family (`yolov8` → `yoloseg` → `yolo26`): adds only
     elementwise/reshape-class ops (SiLU, Concat/Split, Resize, Softmax-DFL,
     TopK) — no attention, no recurrence, fixed shapes.
  3. Transformer representatives (`rtdetr`, `ecdet`, `videomae`, `vitpose`,
     `depth_anything_v2`, `rfdetr*`, `owlv2`, `lgm`, LightGlue matcher) need a
     new attention executor (Softmax+MatMul, LayerNorm, GELU, Gather/TopK,
     int64 paths); `gemma4` excluded (VLM = LLAMACPP column).
  First: ResNet-50 `torchvision_classifier` (E6 representative — bottleneck
  blocks, same family as the merged ResNet18 gate; note the matrix
  representative is ResNet-50, not the ResNet18 the loader currently runs).

### Phase 2 status (2026-11-14)

- [x] Rank 1 — `torchvisionclassifier` ResNet-50 app gate: PASS, top-5 exact
  vs ORT; E6 + E19 + `validation.md` V-7.
- [x] Rank 2 — YOLO conv family: `yolov8` PASS (bicycle 0.89 / dog 0.93 /
  car 0.61) and `yoloseg` PASS (dog 0.87 / bicycle 0.80 / car 0.45 plus
  masks); E2/E5 + E19 + `validation.md` V-8/V-14. The sibling seg graph and
  raw-output parity are recorded in `neuriplo`'s
  `2026-11-14-ggml-yoloseg` packet. `yolo26` is BLOCKED at the converter
  because its `model.23` head uses attention ops (E1/V-12).
- [ ] Rank 3 — transformer representatives: op-level attention/executor
  foundation, a multi-input/INT64 graph executor, and a static-shape
  ONNX-subset frontend landed in `neuriplo` (`2026-11-14-ggml-attention-ops`,
  `2026-11-15-ggml-graph-executor`, `2026-11-15-ggml-onnx-frontend`).
  Fixed-input raw parity is green: 300/300 detections match ORT by label and
  box (max box L1 2.7e-3, max score diff 2.2e-5). The app gate and sorted
  TopK order remain; dynamic batch stays out of scope (V-15).

### Phase 2 out of scope

- GGML VLM (`gemma4`); GPU execution; new CLI flags/presets/schema unless a
  gated model forces one (recorded as a deviation if so).
- Rewriting other backends' cells; touching `neuriplo-tasks` contracts.
