# Feature Plan — GGML matrix coverage

Derived from `requirements.md`. Numbered groups, dependency order, each ending in something observable. Pyramid execution: orchestrator (this session) → two worker subagents (brev live-attempt, docs column) → reviewer pass.

## Group 1 — Foundation (brev + scaffolding)

1. Create cheap brev CPU instance for the attempt:
   `brev create ggml-matrix-coverage --type cpu-e2.4vcpu-16gb` (nebius, 4 vCPU / 16 GB / x86_64, ~$0.14/hr; CPU-only matches `GGMLInfer::setup_backend` which forces CPU). Verify `brev exec ggml-matrix-coverage "nproc && free -g && uname -m"`.
2. On brev: fetch pinned stack (`neuriplo-infer develop` + `versions.env` pins, GGML SDK version owned by `neuriplo`), install `GGML_DIR` SDK per `neuriplo/cmake/DependencyValidation.cmake:457-506` (`include/ggml.h`, `include/ggml-backend.h`, `libggml-base/cpu/blas`).
3. Record baseline: `GGMLInfer.cpp:91` fixed `3x224x224 F32` input, `:128` fixed `[batch,1000]` output, `:142-145` single-input-only, `:153-158` exact-byte-size check, `:179-185` zeros return, `:61-70` CPU-forced. This bounds what the live attempt can possibly show.

## Group 2 — Behavior (live attempt on brev, evidence first)

4. Configure attempt: `cmake -S . -B build-ggml -DDEFAULT_BACKEND=GGML -DCMAKE_BUILD_TYPE=Release`. Capture the exact failure (expected: `cmake/LinkBackend.cmake` has no GGML branch so no backend include dir; `DEFAULT_BACKEND` help string at `CMakeLists.txt:63` omits GGML).
5. Smallest wiring probe that unblocks configure (only if step 4 fails): add the GGML branch to `cmake/LinkBackend.cmake` (include `${neuriplo_SOURCE_DIR}/backends/ggml/src`, no extra link libs beyond what `libneuriplo.so` already carries), plus `DEFAULT_BACKEND` help-string entry. Re-run configure; record pass/fail verbatim. No `app/test`, CI, Dockerfile, or preset changes in this packet.
6. If configure passes, build (`cmake --build build-ggml`) and run the single-shape attempt: classification-shaped input through the GGML path, capturing stdout/stderr, exit code, output shape/values (expected: zeros, proving placeholder). Record exact commands + outputs in `validation.md`. Do not promote any cell on aspiration.
7. Teardown: `brev delete ggml-matrix-coverage` immediately after evidence is recorded. Verify with `brev ls`.

## Group 3 — Contract and documentation (honest column, MGX pattern)

8. `docs/BackendCoverage.md` edits (mirroring `394c2ac`/`511c988`):
   - `Last updated` → `2026-11-13`; no-`ci.yml`-job list += GGML; legend += `` `GGML` = GGML, the independent `GGML` backend in neuriplo, distinct from the `LLAMACPP` (llama.cpp) column ``.
   - Second-table header += `GGML`; every task row `U` (no constructional `N/A`: GGML parses no model bytes, so unlike MGX's ONNX-only scope there is no by-construction exclusion; `gemma4` stays `U`).
   - Preamble paragraph (builder + engine-test green, cells stay `U`, pointer to `E19`); `Scope and caveats` GGML paragraph (upstream tests do not promote task cells; no app build/preset gate until step 5/6 says otherwise, updated to reflect what was actually observed).
   - New `E19 — GGML coverage status`: Slurm jobs 29/30 + `77/77` detail condensed out of `E11`, plus brev attempt evidence (configure log excerpt, run output excerpt, verdict). `E11` keeps LLaMA PASS + one-line `GGML detail: E19` cross-ref.
   - `Environments` += GGML row (upstream env/image, `DEFAULT_BACKEND=GGML`, brev instance type, `ctest 77/77` upstream, app-surface verdict from step 6).
   - `Open items` += GGML app-gate follow-up (real model loading belongs to `neuriplo`). Scoreboard untouched.
9. `CHANGELOG.md` `[Unreleased]` entry: GGML column added as `UNTESTED`/`BLOCKED` with `E19` pointer; no `PASS` claimed.

## Group 4 — Verification

10. Run `validation.md` checks: default build + full test suite still green (this packet must not break the NERT default), `sync_supported_model_types.py --check`, `capabilities_schema_contract`, doc hyperlink check, brev instance confirmed deleted.
11. Reviewer pass (subagent, read-only): matrix claims vs evidence (no cell above its proof), tech-stack/architecture boundary respect, GitFlow shape (spec + code + changelog in one `feature/ggml-matrix-coverage` branch toward `develop`).

## Phase 2 — per-model GGML gates

Dependency order follows `requirements.md` R-8. Rank 1 and 2 are done; the
rest need sibling graph builders before this repo can gate them. Phase 2
supersedes the Phase 1 all-`U` wording in steps 8-9 for the covered cells
(classification, `yolov8`, `yoloseg`).

12. Build the GGML app against the sibling loader branch
    (`neuriplo` `feature/ggml-yolo-family` @ `9ee2bde`) with
    `-DDEFAULT_BACKEND=GGML` and `GGML_DIR`; record the cache provenance.
13. Classification `torchvisionclassifier` (E6): convert
    `resnet50-v1-12.onnx` → NGGL, run `data/dog.jpg`, compare top-5 against
    ORT on identical preprocessing, inspect the render.
14. Detection `yolov8` (E2): export `yolov8n.onnx` (ultralytics, opset 12,
    static 640), convert → NGGL, run `data/dog.jpg` @ 0.4/0.5, compare the
    three detections to the DNN/ExecT/LiteRT/TF gates. Then `yoloseg` (E5):
    export `yolov8n-seg.onnx`, build the sibling seg graph (multi-output +
    cv4/proto), run @ 0.4/0.5/0.5, inspect masks, and record the backend
    parity numbers from `GGMLInferTest`.
15. Record only what the runs prove: E2/E5/E6/E19, Environments, Open items,
    `CHANGELOG.md`, and this packet's `validation.md`. Promote only the three
    covered GGML cells; leave `yolo26`/transformer rows `U`.
16. Next: sibling-side attention work for `yolo26` and the transformer
    representatives, then re-gate here. Do not claim those cells before that
    lands.
