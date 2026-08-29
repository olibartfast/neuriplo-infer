# Tech Stack

Technical boundaries every feature in this repo respects. Read this before
planning a feature; when a discovery changes one of these rules, update this file
in the same branch as the code.

## Canonical sources

These files, not this document, are authoritative for the values they hold.
Never copy a version number out of them into prose:

| Question | File |
|---|---|
| Build requirements, backend options, fetched dependencies | [`CMakeLists.txt`](../CMakeLists.txt) |
| Sibling ref derivation and version loading | [`cmake/versions.cmake`](../cmake/versions.cmake) |
| Concrete sibling pins | [`versions.env`](../versions.env) |
| Released version | [`VERSION`](../VERSION) |
| Upstream TaskFactory model-type inventory | [`docs/generated/supported-model-types.md`](../docs/generated/supported-model-types.md) (generated) |
| The published `--capabilities` contract | [`docs/capabilities.schema.json`](../docs/capabilities.schema.json) |

## Language and build

- C++20, CMake ≥ 3.24, single application target plus a test target.
- OpenCV ≥ 4.6 and glog are system dependencies.
- Siblings (`neuriplo-tasks`, `neuriplo`, `videocapture`, `neuriplo-kserve-client`)
  arrive through `FetchContent`, pinned to concrete `vX.Y.Z` tags in `versions.env`.
  A branch pin (`master`, `develop`) is a release-blocking error —
  `scripts/validate_release_pins.sh` and `release-guard.yml` enforce it.
- Configure-time switches that features must keep working:
  `DEFAULT_BACKEND`, `NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS`,
  `NEURIPLO_INFER_ENABLE_KSERVE`, `NEURIPLO_INFER_ENABLE_GRPC`,
  `NEURIPLO_INFER_ENABLE_KSERVE_TLS`, `ENABLE_APP_TESTS`, `WERROR`,
  `USE_FFMPEG` / `USE_GSTREAMER`.
- A KServe-only build (no local backend) and a local-only build (no KServe) must
  both configure and build. Neither is allowed to become a de facto requirement
  of the other.

## Runtime shape

The layering is fixed; see [`architecture.md`](architecture.md) for the flow
through `CommandLineParser` → `NeuriploInfer` → `InferencePipelineBuilder` →
`CLICommands` → `ResultRenderer`, and for the routing contract in `TaskRouting`.

## Tests and checks

- GoogleTest, discovered by CTest, built only under `-DENABLE_APP_TESTS=ON`.
- Canonical local flow:
  ```bash
  cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build-test
  ctest --test-dir build-test --output-on-failure
  ```
- `clang-format` and `cppcheck` run via `.pre-commit-config.yaml`; `.clang-tidy`
  is checked in. [`scripts/check_code_quality.sh`](../scripts/check_code_quality.sh)
  adds ASan/UBSan and TSan builds when the tools are installed.
- CI: `ci.yml` (build/test plus the generated-docs `--check`), `lint.yml`,
  `branch-policy.yml`, `kserve-integration.yml`, `e2e-example-emulation.yml`,
  `release-check.yml`, `release-guard.yml`.
- `capabilities_schema_contract` validates `--capabilities` output against the
  published schema. A capability may not be advertised before it is routed.

## Workflow

- GitFlow, with `master` as the release branch and `develop` as integration.
  `feature/*` branches from `develop` and merges back by PR; releases and
  hotfixes are the only PRs into `master`, and every merge into `master` is
  back-merged into `develop` immediately. `AGENTS.md` holds the full procedure;
  [`procedures/merge-feature-branch.md`](procedures/merge-feature-branch.md)
  holds the step-by-step merge.
- Agent commits carry a `Co-Authored-By` trailer naming agent, model, and vendor.

## Explicit non-choices

These prevent a feature from expanding the architecture because a familiar
dependency would make one task convenient:

- **No new third-party runtime dependency without maintainer approval.** The
  dependency surface is the siblings, OpenCV, glog, and the backend SDKs.
- **No tensor-shape, dtype, or result-schema semantics in this repo.** Those are
  `neuriplo-tasks` contracts; wiring them through is allowed, redefining them is not.
- **No backend package versions here.** ONNX Runtime, TensorRT, LibTorch,
  OpenVINO, TensorFlow, and CUDA versions are owned by `neuriplo`.
- **No video-backend selection policy here.** `videocapture` owns priority and
  source semantics; this repo only passes the flags through.
- **No hand-editing generated documentation.** The `<!-- SUPPORTED_MODEL_TYPES -->`
  block in `README.md` and `docs/generated/supported-model-types.md` come from
  `scripts/sync_supported_model_types.py`; CI fails on a stale block.
- **No GUI, no server, no daemon.** This is a CLI. Serving belongs to
  `neuriplo-kserve-runtime`; the UI belongs to `neuriplo-ui`.
- **No trunk-based or `main`-only workflow.** Do not propose one.
- **No silent CLI breakage.** Flags are removed or repurposed only as a reviewed
  contract change with a `CHANGELOG.md` entry.
- **No Windows-specific code paths.** Native Windows builds are best-effort with
  a Visual Studio generator and `OPENCV_DNN`; the setup scripts stay Linux-oriented.
