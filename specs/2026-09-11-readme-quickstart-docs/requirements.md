# Feature Requirements — README quickstart and user docs split

Roadmap phase: none — documentation work resolving issues
[#38](https://github.com/olibartfast/neuriplo-infer/issues/38) and
[#39](https://github.com/olibartfast/neuriplo-infer/issues/39), recorded in
[`../roadmap.md`](../roadmap.md) as a done candidate. The issues are closed when
the next release ships this work, not on merge (roadmap Phase 2).
Branch: `feature/readme-quickstart-docs`

## Goal

A first-time user reaches a working detection result from the top-level
`README.md` alone, by copy-pasting commands that were actually run, and learns
which platforms are supported. Everything a first run does not need — the full
CLI reference, KServe, run diagnostics, per-backend deployment, build options,
the model-type inventory — lives in `docs/` and is linked, not repeated.

## In Scope

- `README.md` reduced to: what it is, a Docker quickstart, a build-from-source
  quickstart, platform support, supported task categories, a docs index, known
  limitations, and support.
- `docs/Deployment.md`: platform support, backend matrix (CMake value, setup
  flag, Dockerfile, base image, accelerator, model format), Docker build and run
  per backend (CPU and GPU), headless video runs, the E2E preset script, native
  builds, build options.
- `docs/Usage.md`: CLI synopsis and parameter reference (including flags the
  README never listed: `--no_display`, `--timings_csv`, `--num_frames`,
  `--mmproj`, `--bert_tokenizer_vocab`, `--task_model_version`), examples,
  `--capabilities`, run diagnostics.
- `docs/KserveRuntime.md` absorbs the README's KServe flag section and stops
  pointing at a README section that no longer exists.
- `scripts/sync_supported_model_types.py` stops writing into `README.md`; the
  generated inventory lives only in `docs/generated/supported-model-types.md`.
  Relative links in the upstream block are rewritten to absolute neuriplo-tasks
  URLs so the generated page has no broken links.
- Every rule that names the README marker block is updated: `AGENTS.md`,
  `.cursor/rules/new-task-type-checklist.mdc`, `specs/tech-stack.md`.
- `.dockerignore` excludes `models/` and `environments/`, which the E2E script
  populates on the host and which no Dockerfile copies.
- Known README drift corrected: backend list, Dockerfile count, the
  Windows contradiction, the `--kserve_transport` default.

## Out of Scope

- Issue #44 (datatype propagation) — code change, separate branch.
- The web UI mentioned in the #38 reply — owned by `neuriplo-ui`.
- Making `OPENCV_DNN` load current Ultralytics exports (needs a newer OpenCV than
  Ubuntu's apt 4.6); documented as a limitation instead.
- Rewriting `docs/DependencyManagement.md` beyond pointing its platform note at
  the new page.
- Translating documentation.

## Decisions

- **Quickstart uses ONNX Runtime, not the `OPENCV_DNN` default.** Verified on
  Ubuntu 24.04: apt OpenCV 4.6 rejects YOLO11n (attention `Split`) and YOLOv8n
  (`Unsqueeze`, also with opset 12) at `model_load`. A quickstart built on the
  default backend would fail on its first model.
- **The generated model-type block leaves the README.** It is ~100 lines of
  reference material; the README keeps the manual `## Key Features` category list
  that `AGENTS.md` already requires, plus a link to the generated page.
- **Platform stance comes from the constitution, not new policy:** Ubuntu 24.04 is
  tested (CI, 7 of 8 images); Windows native is best-effort `OPENCV_DNN` only
  (`tech-stack.md`), a non-goal (`mission.md`).

## Constraints and Context

- `tech-stack.md`: no hand-editing generated docs; no silent CLI breakage (docs
  only here — no flag changes).
- `AGENTS.md` hyperlink verification applies to every new link.
- CI `paths-ignore` skips `**.md` / `docs/**`; the sync-script change is `.py`, so
  CI's `--check` step runs on this branch.
- Contract surfaces touched: generated model-type docs (location, not content).

## Open Questions

- None blocking. The maintainer asked for a plug-and-play README with advanced
  features in sub-docs (session 2026-09-11).
