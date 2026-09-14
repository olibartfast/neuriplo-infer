# Feature Validation — README quickstart and user docs split

## Automated

- [x] Generated docs are in sync and the README is no longer a target:
      `python3 scripts/sync_supported_model_types.py --neuriplo-tasks-readme build/_deps/neuriplo-tasks-src/README.md --check`
- [x] The sync script run without `--check` changes nothing in `README.md`.
- [x] No doc, rule, or script still references the README marker block or the
      README *KServe Runtime Parameters* section:
      `grep -rn "SUPPORTED_MODEL_TYPES\|KServe Runtime Parameters" --include=*.md --include=*.mdc --include=*.py .`
      (hits allowed only in `CHANGELOG.md` history and dated spec packets).
- [x] Every relative link in `README.md` and each new or edited `docs/` page
      resolves (`ls`); every absolute link returns 2xx/3xx (`curl -sI`).
- [x] E2E preset dry-run still passes:
      `bash docker_run_inference_e2e_example.sh --preset owlv2 --dry-run`

## Manual

- [x] Docker quickstart, run exactly as written in `README.md` from a clean
      `git archive` of the branch: the image builds, the run exits 0, and the
      annotated output file appears on the host. Record command and output.
- [x] Build-from-source quickstart, run as written (ONNX Runtime backend): the
      build succeeds, the run exits 0, output written.
- [x] Every flag in `docs/Usage.md` exists in `neuriplo-infer --help`, and every
      flag in `--help` is documented.
- [x] Negative evidence for the backend decision is recorded: `OPENCV_DNN`
      failing on YOLO11n / YOLOv8n exports.

## Definition of Done

- [x] Every requirement is implemented or explicitly deferred in `requirements.md`.
- [x] Nothing in *Out of Scope* was implemented anyway.
- [x] Deviations from this file are recorded here, honestly.
- [x] `CHANGELOG.md` describes the change; `../roadmap.md` records it.
- [x] Spec, docs, changelog, and roadmap tell the same story in one branch.

## Results

Run 2026-09-11 on Ubuntu 24.04 x86_64, Docker 29.7.2, RTX 3060 Laptop GPU
(unused — every quickstart run was CPU).

- **Sync check.** `build/_deps/neuriplo-tasks-src` did not exist locally (the
  local build uses the `../neuriplo-tasks` sibling, which is 5 commits past the
  pin). Ran against the pinned tag instead:
  `git -C ../neuriplo-tasks show v0.8.0:README.md > <tmp>` then the script with
  `--neuriplo-tasks-readme <tmp>` → `check passed`. The only change to the
  generated page is the rewritten segmentation-outputs link.
- **Stale references.** The grep returns only this file.
- **Links.** 0 missing relative targets or anchors across `README.md`,
  `docs/Usage.md`, `docs/Deployment.md`, `docs/KserveRuntime.md`,
  `docs/DependencyManagement.md`, and the generated page; 21 absolute URLs return
  200. One (`neuriplo-tasks/.../segmentation/edgecrafter/README.md`) timed out in
  the batch run and returned 200 when retried alone.
- **E2E dry-run.** Needs `--neuriplo-tasks-dir`; without it the script exits 1
  asking for the checkout (as CI provides). With `--neuriplo-tasks-dir
  ../neuriplo-tasks` → exit 0, `Workflow completed for preset 'owlv2'`.
- **Docker quickstart.** `git archive HEAD` into a clean directory;
  `docker build -t neuriplo-infer:onnxruntime -f docker/Dockerfile.onnxruntime .`
  → exit 0, 2.43 GB image. The README `docker run` command, with `yolo11n.onnx`
  in `models/` → exit 0, `Saved processed image to:
  data/output/processed_yolo_local.png`, `run_report.json` `status: success`,
  `samples: 1`. Output files are root-owned; `--user "$(id -u):$(id -g)"` fails
  with `libonnxruntime.so.1: cannot open shared object file` (SDK under
  `/root`), so the docs recommend `chown` instead and the image issue is a
  roadmap candidate.
- **Model export.** Fresh venv, `pip install ultralytics`,
  `yolo export model=yolo11n.pt format=onnx` → exit 0, `yolo11n.onnx` written.
- **Build from source.** `cmake -DDEFAULT_BACKEND=ONNX_RUNTIME` + build → exit 0
  against the ONNX Runtime 1.19.2 already in `~/dependencies` (the setup script
  itself was not re-run; the Docker build exercises the same download).
  `./build/app/neuriplo-infer --type=yolo --weights=yolo11n.onnx
  --source=data/dog.jpg --labels=labels/coco.names` → exit 0, `status: success`.
- **Flags.** Every `--` flag in `--help` appears in `docs/Usage.md` or
  `docs/KserveRuntime.md`; the only documented name absent from `--help` is
  Triton's `--model-control-mode`, cited as a server option.
- **Negative evidence.** Default `OPENCV_DNN` build (apt OpenCV 4.6.0): YOLO11n →
  `[Split]` parse error at `model_load`; YOLOv8n → `[Unsqueeze]` parse error, also
  with `opset=12 simplify=True`. Both with and without `--input_sizes`.

Deviations: the CMake unit-test suite was not run — no C++ or CMake file
changed. The setup script download was not re-run natively (see above).
