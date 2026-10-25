# Feature Requirements — Move app-surface tests from neuriplo-platform

Branch: `feature/move-app-surface-tests`

## Goal

Four integration tests that exercise `neuriplo-infer`'s runtime surface live in
`neuriplo-platform/integration-tests/` and rot when app flags move. Move them
into this repo at `integration-tests/`, re-anchor platform-relative paths, and
delete them from `neuriplo-platform` (with its README matrix and `AGENTS.md`
references updated) in the same move.

## In Scope

Move verbatim, then re-anchor, these four directories:

- `local-inference-smoke/` (runner dry-run half only — see Decisions)
- `kserve-runtime-yolo-e2e/`
- `rfdetr-pose-local-vs-kserve/`
- `kserve-runtime-edgecrafter-e2e/`

Destination: `neuriplo-infer/integration-tests/<name>/`, preserving internal
layout so `Path(__file__).resolve().parents[2]` still resolves to the repo
root and `REPOS_ROOT = ROOT.parent` still resolves to the sibling-checkout
directory. No logic changes beyond path re-anchoring and README rewrites.

Platform side (same move, separate commit in that repo): delete the four dirs,
update `integration-tests/README.md` (Current Tests + Validated Matrix Cases),
drop the `local-inference-smoke/run.py` line from `AGENTS.md` build commands.

This repo side: register the cheap checks in CTest (scaffold/existence and
dry-run-only paths); live paths (runtime binary + model + GPU) stay manual with
documented prerequisites. `CHANGELOG.md` entry. `docs/BackendCoverage.md`
Evidence column may cite the moved tests where they gate a cell.

## Out of Scope

- `kserve-ensemble/`, `failure-modes/`, `openai-generative-serving/`,
  `opencv-boundary-v060/` stay in `neuriplo-platform` (server-side or
  platform-owned scenarios, not app surface).
- Live-run promotion of any cell (GPU/model provisioning is a follow-up).
- `--capabilities`, schema, generated docs, CLI behavior — untouched.
- ADR 0003 stays valid: runners live in owning repos; platform keeps
  orchestration for what remains there.

## Decisions

- Destination `integration-tests/` at repo root (not `app/test/`): keeps every
  script's `parents[2]`/`REPOS_ROOT` computation working verbatim, and mirrors
  the platform layout the READMEs already describe.
- Smoke split: `local-inference-smoke/run.py`'s `versions.yaml` + sibling-tag
  validation is platform orchestration (that file does not exist here; pins
  live in `versions.env`). The moved smoke keeps the app-owned E2E runner
  dry-run half; the cluster-validation half is deleted with the platform copy,
  per the delete-entirely scope answer.
- CTest wiring follows the existing pattern: dry-run/scaffold checks register
  like `docker_run_inference_e2e_owlv2_dry_run`; anything needing a runtime
  binary, model artifact, or GPU is manual-only, like `kserve_integration.sh
  --live`.
- No behavior changes to the moved scripts beyond re-anchoring defaults and
  usage strings. Evidence baselines/reports move as data, unmodified.

## Constraints and Context

- `tech-stack.md`: no backend versions in prose; generated docs untouched
  (`sync_supported_model_types.py --check` must stay green); hyperlinks must
  resolve.
- `check_platform.py` shape rule (README phrases + one executable per dir)
  applies to what remains in platform, not here — but keep the moved dirs
  shape-clean anyway (README + executable entry point each).
- CI `ci.yml` `paths-ignore` covers `**.md`/`docs/**` only: moved `.py`
  files trigger CI, which is intended.
- Cross-repo impact: `neuriplo-platform` deletions + doc updates committed
  there (lands on `main` directly per that repo's rules); this branch holds
  only the neuriplo-infer side.

## Open Questions

- None blocking. Live prerequisites (which runtime build dir, which model
  artifact, GPU or not) are recorded per-test in the moved READMEs as
  found; filling gaps (e.g. missing `rfdetr_keypoint` emulation) stays in
  `docs/BackendCoverage.md` Open items.
