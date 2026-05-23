# Progress Memo

Last updated: 2026-05-23

## Current State

`vision-inference`
- agentic maintenance control-plane docs were merged to `develop`
- CI fixes for topic-branch dependency checks and slower OPENCV_DNN installs were included in that merged work

`vision-core`
- branch pushed: `chore/repo-meta-agentic-maintenance`
- adds repo-local `AGENTS.md` and `REPO_META.yaml`

`neuriplo`
- branch pushed: `chore/repo-meta-agentic-maintenance`
- adds repo-local `AGENTS.md` and `REPO_META.yaml`

`videocapture`
- remote `develop` was stale relative to `master`
- remote `develop` was deleted and recreated from current `master`
- valid PR is `#5`: https://github.com/olibartfast/videocapture/pull/5
- `videocapture#3` is garbage and should be closed
- `videocapture#4` is obsolete and should be closed

## WIP: Quest 3S YOLO QNN Inference Demo

Goal: run an end-to-end YOLOv8n inference example on Meta Quest 3S using ExecuTorch with the Qualcomm QNN backend, targeting the Snapdragon XR2 Gen 2 Hexagon HTP.

Scope:
- standalone on-device demo first; do not wire this into the main application runtime until the device path is proven
- use ExecuTorch QNN backend rather than ONNX Runtime QNN for the first implementation
- target Quest 3S first; OnePlus 10T can follow after the flow is stable
- preserve existing CLI/runtime behavior while this remains an experimental demo path

Known host state:
- `adb` is installed
- Android NDK r26d is installed at `~/dependencies/android-ndk-r26d`
- ExecuTorch QNN AOT pieces appear to exist in `environments/yolo-executorch-export`
- Quest 3S is not yet visible in `adb devices`
- Qualcomm QAIRT/QNN SDK is still required and must be downloaded manually because it is license-gated

Planned phases:
1. Host prerequisites
   - verify `adb devices -l` shows the Quest 3S after developer mode and USB debugging are enabled
   - set `ANDROID_NDK_ROOT=~/dependencies/android-ndk-r26d`
   - install/extract QAIRT/QNN SDK and set `QNN_SDK_ROOT`
   - confirm available HTP architecture libraries, expected to include v73 for Snapdragon XR2 Gen 2
2. AOT export
   - export YOLOv8n from Ultralytics/Torch to an ExecuTorch `.pte`
   - lower supported graph regions to QNN with PTQ INT8 calibration
   - allow documented CPU fallback for unsupported YOLO head/postprocessing pieces
   - write output to `models/e2e/yolov8n_qnn.pte`
3. Cross-compile runner
   - build ExecuTorch `qnn_executor_runner` for Android `arm64-v8a`
   - use the NDK CMake toolchain from `~/dependencies/android-ndk-r26d/build/cmake/android.toolchain.cmake`
   - link/package the QNN ExecuTorch backend and required QAIRT shared libraries
4. Deploy and run on Quest
   - push runner, `.pte`, input tensor, and required QNN libraries to `/data/local/tmp/vision-inference-qnn`
   - set `LD_LIBRARY_PATH` and `ADSP_LIBRARY_PATH` for the pushed library layout
   - run inference through `adb shell`
   - pull output tensors and run host-side YOLO postprocessing/NMS and visualization
5. Repo scripts and documentation
   - add `scripts/setup_qnn.sh` for prerequisite validation
   - add `scripts/export_yolo_qnn.sh` for AOT export
   - add `scripts/run_yolo_qnn_device.sh` for cross-compile, deploy, and run
   - optionally add `docker/Dockerfile.qnn` for a reproducible host build/export environment
   - add `docs/quest-qnn-inference.md` with prerequisites, export, build, deploy, run, and troubleshooting notes

Current blockers:
- Quest 3S must be connected and authorized for `adb`
- QAIRT/QNN SDK must be downloaded from Qualcomm and its extracted path provided

## Valid Branches / PRs

Keep:
- `vision-core`: `chore/repo-meta-agentic-maintenance`
- `neuriplo`: `chore/repo-meta-agentic-maintenance`
- `videocapture`: `chore/repo-meta-agentic-maintenance-v2`
- `videocapture` PR `#5`

Ignore or close:
- `videocapture` PR `#3`
- `videocapture` PR `#4`

## Release Status

- coordinated `0.2.0` release prep has **not** been done yet
- no new release tags should be created until release bumps/changelogs are finalized on the intended branches
- policy remains: release tags only when closing on `master`

## Next Step When Resuming

1. finish or merge the sibling metadata/versioning PRs
2. verify clean branch state across all four repos
3. prepare coordinated `0.2.0` release bumps
4. open release PRs to `master` only when ready
