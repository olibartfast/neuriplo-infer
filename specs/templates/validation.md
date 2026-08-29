# Feature Validation — <feature name>

> Copy to `specs/YYYY-MM-DD-<feature-name>/validation.md`. Write this *before*
> implementation, so the criteria cannot be fitted to whatever gets built.
> Execute it from this file at stage 6 — do not summarize it from memory.

## Automated

- [ ] Default build configures and builds:
      `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [ ] Test build and full suite pass:
      `cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-test && ctest --test-dir build-test --output-on-failure`
- [ ] Focused tests cover the success path *and* the failure path this feature adds.
- [ ] `pre-commit run --all-files` (clang-format, cppcheck) is clean.
- [ ] Generated docs are in sync: `python3 scripts/sync_supported_model_types.py --check`
- [ ] `--capabilities` still validates (`capabilities_schema_contract`).
- [ ] Build variants this feature could break still configure — list the ones that
      apply (KServe-only, local-only, `-DNEURIPLO_INFER_ENABLE_GRPC=OFF`, `-DWERROR=ON`).

## Manual

- [ ] The primary invocation behaves as `requirements.md` specifies — record the
      exact command and the observed output.
- [ ] Invalid, empty, and unsupported-combination inputs fail fast with an error
      naming the flag and the way out.
- [ ] Relevant E2E preset still runs:
      `bash docker_run_inference_e2e_example.sh --preset <preset> --dry-run`
- [ ] Every hyperlink added to docs resolves (`ls` for relative, `curl -sI` for absolute).

## Definition of Done

- [ ] Every requirement is implemented or explicitly deferred in `requirements.md`.
- [ ] Nothing in *Out of Scope* was implemented anyway.
- [ ] Deviations from this file are recorded here, honestly, with what was run instead.
- [ ] `CHANGELOG.md` describes the user-visible change; `../roadmap.md` phase
      status is updated.
- [ ] Spec, code, changelog, and roadmap tell the same story in one branch.
