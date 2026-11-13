# Feature Plan - TensorFlow GPU and remote TPU

**Packet:** `2026-11-09-tensorflow-gpu-remote-tpu`
**Branch:** `feature/tensorflow-gpu-remote-tpu`

Each group ends in a runnable or inspectable result. Group 1 depends on the
`neuriplo` GPU image name and the `neuriplo-tasks` export command.

## Group 1 - TensorFlow e2e plumbing

- [T-1] Add `tensorflow` and `tensorflow-gpu` to the e2e backend validation and
  help text.
- [T-2] Extend `MODEL_ARTIFACT` handling to accept a SavedModel directory, not
  only a file suffix. Mount the directory under `/weights/<name>` and pass that
  path to `--weights`.
- [T-3] Add a `tensorflow_classifier` preset:
  `MODEL_TYPE=tensorflow-classifier`, `MODEL_BASENAME=tensorflow-classifier`,
  `SOURCE_IN_CONTAINER=/app/data/dog.jpg`,
  `LABELS_IN_CONTAINER=/labels/imagenet_labels.txt`.
- [T-4] Add the export command using
  `export/classification/tensorflow/export_tf_saved_model_classifier.py`.
- [T-5] For `tensorflow-gpu`, add `--gpus=all` to `DOCKER_ARGS` and
  `--use-gpu` to `RUNTIME_EXTRA_ARGS`. For `tensorflow`, add neither.
- [T-6] Add the preset to `list_presets`.

## Group 2 - TPU harness

- [T-7] Create `integration-tests/tpu-kserve/README.md` with TPU VM
  prerequisites, service contract, dry-run command, and live command.
- [T-8] Create `integration-tests/tpu-kserve/run.py`:
  - `--dry-run` validates required files and prints the CLI command without
    network access.
  - `--live` invokes `neuriplo-infer` against `--endpoint`, checks exit status
    and output artifact, and records the KServe response/run report.
  - A numeric mode compares the served top-5 classes/scores with the CPU
    baseline.
- [T-9] Add `baselines/` and `evidence.yaml` with a `skipped` live check until a
  TPU run is attested.
- [T-10] Register the dry-run in `app/test/CMakeLists.txt` using the existing
  evidence-scaffold pattern.

## Group 3 - Tests and CI

- [T-11] Extend `app/test/test_docker_run_inference_e2e_example.sh` with:
  - A temporary `imagenet_labels.txt` fixture for the classifier preset.
  - CPU TensorFlow dry-run: no `--gpus`, no `--use-gpu`, directory weights path.
  - GPU TensorFlow dry-run: `--gpus=all`, `--use-gpu`,
    `neuriplo-infer:tensorflow-gpu`.
- [T-12] Add `tensorflow_classifier` to the
  `.github/workflows/e2e-example-emulation.yml` matrix.
- [T-13] Add a focused dry-run job or step for `--backend tensorflow-gpu`; do not
  add a full matrix axis unless the existing workflow shape supports it cleanly.
- [T-14] Keep `ctest` green without a GPU or TPU.

## Group 4 - Documentation and evidence

- [T-15] Update `docs/Deployment.md`:
  - TensorFlow GPU image build and run command.
  - TPU VM/service topology and the client invocation.
- [T-16] Update `docs/Usage.md` for `--use-gpu` on TensorFlow and the TPU KServe
  invocation.
- [T-17] Update `docs/KserveCompatibility.md` with the TPU custom server row and
  its transport coverage.
- [T-18] Update `docs/BackendCoverage.md` with an accelerator evidence section
  and links to the packet/evidence. Do not promote a cell before the live run.
- [T-19] Add the `CHANGELOG.md` entry only after evidence exists.
- [T-20] Regenerate or check generated docs; no supported-model-types change is
  expected.

## Acceptance

The packet is done when the TensorFlow CPU and GPU e2e commands are well-formed
in dry-run, the TPU harness dry-run is registered and green, the app test suite
passes without accelerators, and any live GPU/TPU run is recorded with exact
commands and results.
