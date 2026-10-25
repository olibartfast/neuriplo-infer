# Feature Validation — Backend coverage matrix

Written before implementation; executed at stage 6 from this file.

## Automated

- [ ] Default build configures and builds:
      `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [ ] Test build and full suite pass:
      `cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-test && ctest --test-dir build-test --output-on-failure`
- [ ] `python3 scripts/sync_supported_model_types.py --neuriplo-tasks-readme build-test/_deps/neuriplo-tasks-src/README.md --check` is clean (generated docs untouched).
- [ ] `capabilities_schema_contract` passes inside the `ctest` run above.
- [ ] E2E preset dry-runs for every representative preset in the matrix pass:
      `bash docker_run_inference_e2e_example.sh --preset <p> --neuriplo-tasks-dir /tmp/neuriplo-tasks-placeholder --dry-run`
      for `rtdetrv4 owlv2 torchvision_classifier yoloseg yolov8_executorch yolo26s_tflite edgecrafter_det edgecrafter_seg edgecrafter_pose raft vitpose rfdetr_keypoint depth_anything_v2 videomae gemma4`.
- [ ] `pre-commit run --all-files` clean, if available (else note skip).

## Manual

- [ ] Every `PASS` cell has a recorded live output cited in Evidence (detections, labels, or round-trip log); no `PASS` rests on a build or dry-run alone.
- [ ] Every hyperlink added to docs resolves: `ls` for relative paths
      (`docs/BackendCoverage.md`, `docs/Deployment.md`, `docs/KserveCompatibility.md`),
      `curl -sI` for the absolute NERT + sibling links.
- [ ] `README.md` docs table and `docs/Deployment.md` backends section each link the new matrix once.
- [ ] `rfdetr_keypoint` emulation gap is recorded in Open items, not silently fixed or hidden.

## Definition of Done

- [ ] Every requirement is implemented or explicitly deferred in `requirements.md`.
- [ ] Nothing in *Out of Scope* was implemented anyway (no new presets, CI jobs, capability changes).
- [ ] Deviations from this file are recorded here with what ran instead.
- [ ] `CHANGELOG.md` describes the user-visible change; `specs/roadmap.md` untouched (Phase 5 still open, referenced only).
- [ ] Spec, doc, changelog tell the same story in one branch.

## Results — 2026-10-23

Automated: default build OK; test build OK; `ctest` 177/177 pass
(includes `capabilities_schema_contract`, `docker_run_inference_e2e_owlv2_dry_run`,
KServe dry-runs); `sync_supported_model_types.py --check` clean; all 15
presets `--dry-run` OK (incl. `rfdetr_keypoint`, confirming the Open-item gap:
script supports it, emulation matrix does not).
Manual: NERT link `curl -sI` 200; relative links `ls`-verified; README +
Deployment link the matrix once each.
Deviation: `pre-commit` not installed on this runner — skipped. Change is
docs/markdown + packet only (no C++), so clang-format/cppcheck have nothing to
cover; CI `lint.yml` remains the gate.

## Results — post-merge re-validation (develop @ 6951525)

`git merge origin/develop` was a no-op (branch already at tip). Re-ran on the
merged tree: test build OK, `ctest` 177/177 pass, sync `--check` clean, all 15
script presets `--dry-run` green. Emulation matrix still lists 14 presets
(`rfdetr_keypoint` still absent) — Open item 2 stands. No cell changed status.
