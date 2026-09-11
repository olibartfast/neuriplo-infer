# Feature Validation — Model input datatypes reach preprocessing

## Automated — neuriplo-tasks

- [ ] `find src include tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror`
- [ ] `cppcheck --enable=warning --std=c++17 --error-exitcode=1 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=*:3rdparty/stb/* -I include src/`
- [ ] clang-tidy-18 over `src/` with `-DNEURIPLO_TASKS_WITH_OPENCV=ON` compile commands
- [ ] `-DWERROR=ON -DBUILD_TESTS=ON` builds; `ctest` passes (with and without OpenCV)
- [ ] New tests: `UInt8` raw output (NCHW, NHWC, letterbox), `Float32` byte-identical,
      `Int32` rejected, `UInt8` on RAFT / video rejected, TensorFlow classifier unchanged.

## Automated — neuriplo-infer

- [ ] Format check, cppcheck, clang-tidy exactly as `lint.yml`
- [ ] `-DWERROR=ON -DENABLE_APP_TESTS=ON` build against the neuriplo-tasks branch; all tests pass
- [ ] KServe-only build (`-DNEURIPLO_INFER_ENABLE_LOCAL_BACKENDS=OFF`) configures and builds
- [ ] New tests: metadata datatype propagation; byte-count guard accepts FP32 / UINT8 /
      INT64 size / dynamic encoded input and rejects INT8 / BOOL / UINT8 / FP16 mislabels
      before `infer()`; setup-time rejection names input and datatype; `encoded-image` exempt.

## Manual — end to end

- [ ] `neuriplo-kserve-runtime` built against neuriplo v0.9.1 serves `UINT8`, `INT8`,
      `BOOL` image-input classifiers.
- [ ] `UINT8`: `neuriplo-infer` run exits 0 with a classification result.
- [ ] `INT8`, `BOOL`: run exits non-zero at setup; message names the input and datatype;
      no inference request reaches the runtime.
- [ ] A local ONNX Runtime run of an FP32 model is unchanged (regression).

## Definition of Done

- [ ] Every requirement implemented or explicitly deferred.
- [ ] Nothing in *Out of Scope* implemented.
- [ ] Deviations recorded below.
- [ ] Both `CHANGELOG.md` files updated; pins moved to tags; lint green before every push.

## Results

Run 2026-09-11, Ubuntu 24.04, neuriplo v0.9.1, neuriplo-tasks
`feature/model-input-datatypes` (worktree off `origin/develop` 73295af).

- **neuriplo-infer lint.** Synced tree built with
  `-DFETCHCONTENT_SOURCE_DIR_NEURIPLO-TASKS=<neuriplo-tasks branch>`:
  clang-format PASS, cppcheck PASS, `-DWERROR=ON` build PASS, CTest 119/119
  (108 before: 10 new unit tests plus `kserve_input_datatypes_e2e_dry_run`),
  KServe-only (`-DNEURIPLO_INFER_ENABLE_LOCAL_BACKENDS=OFF`) `-DWERROR=ON` build
  PASS. clang-tidy exit 0; its one finding in new code
  (`performance-inefficient-string-concatenation`, `ModelInputTypes.cpp`) was
  fixed and that file re-checked clean. The five remaining findings in
  `InferencePipeline.{hpp,cpp}` are identical on `develop` (checked against the
  docs-branch clang-tidy log) and were left alone.
- **End to end.** `neuriplo-kserve-runtime` (`origin/develop` f4e4bff, preset
  `real-onnx`) built against neuriplo v0.9.1, serving four 1×3×32×32 image-input
  classifiers (`Cast` → `ReduceMean`) in repository mode; metadata advertises
  `UINT8`, `INT8`, `BOOL`, `FP32`. `neuriplo-infer --type=torchvision-classifier
  --kserve_transport=http` against each, runtime request counters read before
  and after:

  | Binary | Model | Exit | Outcome | Requests accepted |
  |--------|-------|------|---------|-------------------|
  | before (develop + v0.9.1 pin) | `UINT8` | 1 | HTTP 400 `input data element count mismatch for input: images` | 0 |
  | after | `UINT8` | 0 | success | 1 |
  | after | `INT8` | 1 | `model_load`: `model input 'images' is INT8, but image preprocessing produces FP32 or UINT8 tensors; …` | 0 |
  | after | `BOOL` | 1 | `model_load`: same, naming `BOOL` | 0 |
  | after | `FP32` | 0 | success | 1 |

- **Local regression.** Same binary, OpenCV DNN, `cls_fp32` model,
  `--input_sizes='3,32,32'`: exit 0, run report `success`, 1 sample.
- **Repeatable harness.** The runs above are now
  `app/test/kserve_input_datatypes_e2e.sh`: `bash -n` PASS, ShellCheck 0.11.0
  PASS, `--dry-run` PASS (registered in CTest as
  `kserve_input_datatypes_e2e_dry_run`), and `--require-live` against the same
  runtime and binary PASS on all four cases (UINT8 and FP32 ran with 1 accepted
  request each; INT8 and BOOL refused at setup with 0 requests).
- **neuriplo-tasks.** clang-format PASS, cppcheck PASS, `-DWERROR=ON` builds with
  and without OpenCV PASS, CTest 30/30 and 29/29 (new `test_image_input_type`
  included), Valgrind on the new test binary clean.
- **Release pin (2026-09-11).** PR #11 was merged and released as neuriplo-tasks
  `v0.8.2` (tag + GitHub release), and `versions.env` now pins
  `NEURIPLO_TASKS_VERSION=v0.8.2`. The full tree was rebuilt from a clean
  worktree against that tag (`-DWERROR=ON`, `FETCHCONTENT_SOURCE_DIR_NEURIPLO-TASKS`
  pointing at a `v0.8.2` checkout): format PASS, cppcheck PASS, build PASS,
  CTest 119/119, `kserve_input_datatypes_e2e --dry-run` PASS. This is the pin the
  PR ships; the earlier runs above used the pre-release branch and are superseded
  for the committed state.

## Deviations

- Plan step 10 also promised a builder-level test (a fake engine driving
  `setupTask` / `buildModelInfo`). The datatype mapping, the rejection, and the
  `encoded-image` exemption are covered directly by `test_ModelInputTypes.cpp`;
  the builder wiring is exercised end-to-end by
  `app/test/kserve_input_datatypes_e2e.sh --require-live` rather than a unit
  test, because `buildModelInfo` / `inputDatatypes` are translation-unit-local
  and exposing them only for a test was not judged worth the API change.
- The live e2e (`--require-live`) stays a manual / `workflow_dispatch` check; CI
  runs the dry-run only, so no serving runtime is needed on every PR.
- The `v0.8.1` detector-normalization change the pin pulls in changes detector
  output on RT-DETR / RT-DETRv2 / D-FINE / DEIM; it is called out in
  `CHANGELOG.md` but not re-validated here (out of #44 scope).
