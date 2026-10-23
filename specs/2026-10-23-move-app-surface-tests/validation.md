# Feature Validation — Move app-surface tests from neuriplo-platform

Written before implementation; executed from this file at stage 6.

## Automated

- [ ] Default build configures and builds:
      `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [ ] Test build and full suite pass:
      `cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-test && ctest --test-dir build-test --output-on-failure`
- [ ] Newly registered CTest entries pass (scaffold + dry-run checks).
- [ ] `python3 scripts/sync_supported_model_types.py --neuriplo-tasks-readme <path> --check` clean.
- [ ] Moved dry-run-capable scripts pass from their new location (smoke runner
      half; rfdetr scaffold `run.py`; any `--dry-run` path the moved suites offer).
- [ ] Platform side: `scripts/check_platform.py` passes after deletions.

## Manual

- [ ] Every moved file accounted for: `diff -r` of each platform dir against
      its new home shows only intended re-anchoring deltas.
- [ ] No live test was wired into routine CI (runtime binary / model / GPU
      prerequisites stay manual, READMEs say so).
- [ ] Every hyperlink added or rewritten resolves (`ls` relative, `curl -sI` absolute).
- [ ] Platform `integration-tests/README.md` no longer lists moved tests;
      `AGENTS.md` no longer references the deleted smoke path.

## Definition of Done

- [ ] Four dirs live under `integration-tests/` and run from there.
- [ ] Platform no longer contains them; its docs do not reference them.
- [ ] Nothing in *Out of Scope* was moved or changed.
- [ ] Deviations recorded here with what ran instead.
- [ ] `CHANGELOG.md` describes the move; spec and code tell the same story.

## Results — 2026-10-23

Automated: default build OK; test build OK; `ctest` 179/179 pass (177 + 2 new:
`local_inference_smoke_runner`, `rfdetr_pose_evidence_scaffold`);
`sync_supported_model_types.py --check` clean; smoke runner and rfdetr
scaffold pass from their new location; platform `scripts/check_platform.py`
passes after deletions.
Manual: `diff -r` of each platform dir vs its new home shows only intended
re-anchoring deltas (owning blocks, `From neuriplo-platform` usage lines,
`versions.yaml`/`versions.env` refs, repo-relative artifact paths); all
`.py` suites are byte-identical. No live test was wired into routine CI.
Relative links `ls`-verified.
Deviations:
- `pre-commit` not installed on this runner — skipped (change is scripts/docs
  + one CMake test block; CI `lint.yml` remains the gate).
- Absolute `blob/develop` / `tree/develop` URLs in the platform edits 404
  until this branch merges to `develop`. Land the neuriplo-infer PR first;
  the platform commit references the post-merge locations.
- Platform side: `generate_compat_report.py` stays in `neuriplo-platform`
  (shared with `kserve-ensemble` / `opencv-boundary`, which stay); the moved
  rfdetr README says so. `versions.yaml` compat sets keep history in
  `notes:` while `evidence:`/moved baseline entries were dropped (the checker
  requires in-repo files).
