# Feature Validation - TensorFlow GPU and remote TPU

Defined before implementation. A check that needs a GPU or TPU is manual or
scheduled; routine CI must stay accelerator-free.

## Automated

- [x] [V-1] -> [R-1], [R-2]: Default build configures and builds:
      `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`.
- [x] [V-2] -> [R-1], [R-2], [R-8]: Test build and full suite pass:
      `cmake -S . -B build-test -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release`
      then `cmake --build build-test` and
      `ctest --test-dir build-test --output-on-failure`.
- [x] [V-3] -> [R-3]: `bash docker_run_inference_e2e_example.sh --preset tensorflow_classifier --dry-run`
      prints the SavedModel export, the directory mount, and `--weights` with a
      directory path; it contains no `--gpus` or `--use-gpu`.
- [x] [V-4] -> [R-3]: The same preset with `--backend tensorflow-gpu --dry-run`
      contains `--gpus=all`, `--use-gpu`, and
      `neuriplo-infer:tensorflow-gpu`; the CPU dry-run does not.
- [x] [V-5] -> [R-4]: `--list-presets` includes `tensorflow_classifier`.
- [x] [V-6] -> [R-5]: `python3 integration-tests/tpu-kserve/run.py --dry-run`
      passes without network access; the CTest registration is green.
- [x] [V-7] -> [R-7]: `--capabilities` validates against
      `docs/capabilities.schema.json`; no schema change is expected.
- [x] [V-8] -> [R-7]: Generated docs checks pass:
      `python3 scripts/sync_supported_model_types.py --check`.
- [ ] [V-9] -> [R-8]: `pre-commit run --all-files` is clean.
- [x] [V-10] -> [R-3], [R-4]: The e2e emulation workflow dry-run matrix includes
      the new preset, and the focused GPU backend dry-run passes.

## Manual

- [ ] [M-1] -> [R-1]: On a GPU host, build/pull
      `neuriplo-infer:tensorflow-gpu`, run the classifier with `--use-gpu`, and
      record `nvidia-smi`, image digest, command, exit status, output path, and
      top-5 classes/scores.
- [ ] [M-2] -> [R-2]: Run the CPU TensorFlow image on the same artifact and
      compare the top-5 output. Record the differences.
- [ ] [M-3] -> [R-4]: On a Cloud TPU VM, start/point to the KServe V2 service
      and run `neuriplo-infer --kserve_endpoint ... --kserve_model_name ...`.
      Record TPU type, runtime, endpoint, command, exit status, output path, and
      top-5 classes/scores.
- [ ] [M-4] -> [R-4]: Compare TPU and CPU top-5 results for the same input;
      record the tolerance and any differences.
- [ ] [M-5] -> [R-6]: Read `docs/Deployment.md` and
      `docs/KserveCompatibility.md` as a new user. Confirm the GPU and TPU
      sections state prerequisites, costs, and what is not validated.
- [ ] [M-6] -> [R-6]: Verify every added relative link resolves and every
      absolute URL returns HTTP 200.

## Acceptance Commands

```bash
cmake -S . -B build-test \
  -DENABLE_APP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-test
ctest --test-dir build-test --output-on-failure

bash docker_run_inference_e2e_example.sh \
  --preset tensorflow_classifier --dry-run

bash docker_run_inference_e2e_example.sh \
  --preset tensorflow_classifier --backend tensorflow-gpu --dry-run

python3 integration-tests/tpu-kserve/run.py --dry-run
```

The live GPU/TPU commands are recorded in the evidence table once the hosts and
image tags are known.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | Build | pass | 2026-11-09 | fresh `build-accel-test` configure + build succeeded; stale `build-test` FetchContent cache avoided |
| V-2 | App tests | pass | 2026-11-09 | `ctest --test-dir build-accel-test --output-on-failure`: 197/197 passed |
| V-3 | CPU TensorFlow dry-run | pass | 2026-11-09 | `bash docker_run_inference_e2e_example.sh --preset tensorflow_classifier --dry-run`; output has no `--gpus`/`--use-gpu`; live GPU not claimed |
| V-4 | GPU TensorFlow dry-run | pass | 2026-11-09 | `... --preset tensorflow_classifier --backend tensorflow-gpu --dry-run`; shows `--gpus=all`, `--use-gpu`, `neuriplo-infer:tensorflow-gpu`; focused dry-run, live GPU not claimed |
| V-5 | Preset advertised | pass | 2026-11-09 | `--list-presets` includes `tensorflow_classifier` |
| V-6 | TPU dry-run | pass | 2026-11-09 | `python3 integration-tests/tpu-kserve/run.py --dry-run` (no network) and CTest `tpu_kserve_dry_run` registered/green; live TPU not claimed. GPU app image note: the build path in `docker/Dockerfile.libtensorflow` (`TF_BASE_IMAGE`/`TF_GPU`) is defined but was not built on this host (no Docker/GPU); live GPU remains unexecuted |
| V-7 | Capabilities schema | pass | 2026-11-09 | `capabilities_schema_contract` passed in the 197/197 suite; no schema change |
| V-8 | Generated docs | pass | 2026-11-09 | `python3 scripts/sync_supported_model_types.py --neuriplo-tasks-readme ../neuriplo-tasks/README.md --check` exit 0 |
| V-9 | Format/lint | pass (partial) | 2026-11-09 | `git diff --check` clean; `bash` dry-run and `test_docker_run_inference_e2e_example.sh` pass; `pre-commit` binary not installed on this host so full `pre-commit run --all-files` not executed |
| V-10 | E2E emulation matrix | pass | 2026-11-09 | `e2e-example-emulation.yml` lists `tensorflow_classifier` and has focused `tensorflow-gpu` dry-run job asserting `neuriplo-infer:tensorflow-gpu` and workflow completion |
| M-1 | GPU live | not executed | 2026-11-09 | needs GPU host; no GPU image published/digest |
| M-2 | CPU/GPU compare | not executed | 2026-11-09 | needs GPU host and same SavedModel artifact |
| M-3 | TPU live | not executed | 2026-11-09 | needs Cloud TPU VM |
| M-4 | TPU/CPU compare | not executed | 2026-11-09 | needs M-3; tolerance procedure documented in tpu-kserve README |
| M-5 | New-user docs read | not executed | 2026-11-09 | docs read not performed |
| M-6 | Link/anchor check | pass | 2026-11-09 | new relative links resolve; new anchors present (`BackendCoverage.md#ev-accelerator`, `#tensorflow-gpu-and-remote-tpu-not-yet-validated`); no new absolute URLs added |

Deviation: this pass did not publish a GPU image or record a digest, ran no
live GPU/TPU inference, and left `CHANGELOG.md` untouched per T-19. The GPU
app image build path is defined in `docker/Dockerfile.libtensorflow`
(`TF_BASE_IMAGE`/`TF_GPU`) but was not built on this host (no Docker/GPU);
live GPU remains unexecuted. `pre-commit` is not installed on this host, so
V-9 is partial (diff-check + shell tests only).

## Definition of Done

- [ ] Every requirement is implemented or explicitly deferred.
- [ ] Nothing in Out of Scope was implemented anyway.
- [ ] Deviations from this file are recorded here, including unrun checks.
- [ ] `CHANGELOG.md` describes the user-visible change after evidence exists.
- [ ] `../roadmap.md` has the accelerator coverage entry when the packet merges.
- [ ] Spec, code, docs, and evidence tell the same story in one branch.
