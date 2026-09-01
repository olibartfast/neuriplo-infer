# Feature Validation — OpenCV-free application layer

Written before implementation, so the criteria cannot be fitted to whatever gets
built. Execute from this file at Group 8 — do not summarize it from memory.

The central claim is falsifiable and must be tested as such: **the application
builds and runs on a machine with no OpenCV installed.** Every other check is
secondary to that one.

## The decisive check

- [ ] On a container with **no OpenCV package present at all**, a non-OpenCV
      backend configures, builds, and runs:
      ```bash
      docker build -f docker/Dockerfile.onnxruntime -t neuriplo-infer:nocv .
      docker run --rm neuriplo-infer:nocv --capabilities   # must succeed
      ```
      This is the only check that cannot pass by accident. A build on a
      developer machine proves nothing, because OpenCV is installed there and
      a stray `find_package` would silently succeed.
- [ ] `ldd` on the built binary lists **no `libopencv_*`** for
      `ONNX_RUNTIME`, `TENSORRT`, `LIBTORCH`, and KServe-only builds — and no
      `libSDL2` unless `NEURIPLO_INFER_WITH_DISPLAY=ON`.
- [ ] A `NEURIPLO_INFER_WITH_DISPLAY=ON` build opens a window and ends on `q`
      or Esc, on a real video, with the overlay drawn — the feature this
      exception was granted for actually works.
- [ ] The **same** `NEURIPLO_INFER_WITH_DISPLAY=ON` binary, run with `DISPLAY`
      unset, processes the whole video to the end without a window and without
      failing — `env -u DISPLAY ./neuriplo-infer --source <video> …`. A display
      build that dies on a headless host is the predictable way this flag
      becomes a support burden.
- [ ] `ldd` on an `OPENCV_DNN` build **does** list OpenCV, reached only through
      `neuriplo` — confirming the dependency moved rather than vanished.
- [ ] `grep -rn "cv::\|opencv2" app/` returns nothing outside whatever the
      display decision (Group 6) deliberately keeps.

## Automated

- [ ] Default build configures and builds:
      `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [ ] Test build and full suite pass:
      `cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-test && ctest --test-dir build-test --output-on-failure`
- [ ] `ctest` is run **serially as well as in parallel**. Six tests currently
      fail only under `-j4` because they share `data/output/` writes; that
      pre-existing flakiness must not be mistaken for a regression from this
      work, nor hide one.
- [ ] `neuriplo-tasks` passes in **both** configurations — `build-ocv` (which
      carries the differential render tests) and `build-no-ocv` (which proves
      the drawing layer itself needs no OpenCV).
- [ ] Differential render tests: for `line`, `rectangle` (stroked and filled),
      `circle`, `polyline` (open and closed), `blend`, `fillMasked`, and
      `applyColormap(Turbo)`, `vision::draw` agrees with OpenCV to
      **|Δ| ≤ 8 per channel on ≥ 99% of pixels** for anti-aliased strokes and
      **exactly** for un-anti-aliased fills.
- [ ] `measureText` agrees with `cv::getTextSize` to **≤ 1 px** in width,
      height, and baseline for the SIMPLEX and DUPLEX fonts at scales 0.8 and
      1.0 with thickness 2 — the values `ResultRenderer` and `utils::draw_label`
      actually use — across ASCII 32–126 and a set of representative labels.
- [ ] `--help` output is **byte-identical** before and after the CLI parser
      replacement (Group 5). Capture it on `develop` first.
- [ ] `pre-commit run --all-files` (clang-format, cppcheck) is clean.
- [ ] `python3 scripts/sync_supported_model_types.py --check` passes.
- [ ] `capabilities_cli_contract` and `capabilities_schema_contract` pass, with
      **no `schema_version` bump** — this feature must not touch the published
      contract.
- [ ] Build variants still configure: KServe-only
      (`-DNEURIPLO_INFER_ENABLE_LOCAL_BACKENDS=OFF`), local-only
      (`-DNEURIPLO_INFER_ENABLE_KSERVE=OFF`), `-DNEURIPLO_INFER_ENABLE_GRPC=OFF`,
      `-DWERROR=ON`, and `-DUSE_FFMPEG=ON`.

## Manual

- [ ] **Visual regression, one image per task type.** For detection, open-vocab
      detection, classification, video classification, instance segmentation
      (mask path *and* polygon path), pose, depth, and optical flow: run the same
      model on the same input on `develop` and on this branch, save both
      outputs, and compare them side by side. Record the pairs. Boxes,
      skeletons, masks, colormaps, and labels must be in the same places, the
      same colours, and the same thicknesses — the mask colour draw uses
      `std::rand()` and will differ in hue, which is expected and is the only
      accepted difference.
- [ ] **Label text is not clipped** on a long class name and a short one, at
      an image edge and at the top-left corner, where `draw_label` clamps.
      This is the failure mode a wrong `measureText` produces, and it does not
      show up in a pixel diff of a short label.
- [ ] The primary invocation behaves as `requirements.md` specifies — record the
      exact command and observed output.
- [ ] Invalid, empty, and unsupported-combination inputs still fail fast with an
      error naming the flag and the way out.
- [ ] `bash docker_run_inference_e2e_example.sh --preset <preset> --dry-run` runs.
- [ ] Image size delta recorded for each Dockerfile touched, before and after.
- [ ] Every hyperlink added to docs resolves.

## Definition of Done

- [ ] Every requirement is implemented or explicitly deferred in `requirements.md`.
- [ ] Nothing in *Out of Scope* was implemented anyway — in particular, no
      rendering *policy* changed: no new colours, no re-laid-out labels, no
      "improved" skeleton.
- [ ] **No headless image gained SDL2.** For every `Dockerfile` built without
      `NEURIPLO_INFER_WITH_DISPLAY`, the measured image-size delta versus today
      is negative, and neither `libsdl2` nor its GL stack (`libgl1`,
      `mesa-libgallium`, `libllvm20`) appears in `dpkg -l`. SDL2's closure is
      +237 MB; leaking it into a serving container would cancel most of this
      feature's benefit while every build still looked correct.
- [ ] A default build does **not** call `find_package(SDL2)` — grep the
      configure log. An optional dependency that is always discovered is a
      required one with extra steps.
- [ ] `neuriplo-tasks` is released and pinned to a concrete `vX.Y.Z` tag in
      `versions.env` before this branch is released
      (`scripts/validate_release_pins.sh` passes).
- [ ] `specs/tech-stack.md` no longer claims OpenCV is a system dependency.
- [ ] Deviations from this file are recorded here, honestly, with what was run
      instead.
- [ ] `CHANGELOG.md` describes the user-visible change — including the display
      change, which is a CLI contract change.
- [ ] Spec, code, changelog, and roadmap tell the same story in one branch.
