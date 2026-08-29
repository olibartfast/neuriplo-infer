# Feature Validation — KServe model repository through the CLI

Written before implementation. Execute this file at stage 6; do not summarize it
from memory. Record deviations in *Deviations* below.

## Automated

- [ ] `versions.env` pins `NEURIPLO_KSERVE_CLIENT_VERSION` to the release carrying
      Phase 3's typed error, and `scripts/validate_release_pins.sh` accepts it.
- [ ] Default build:
      `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [ ] Test build and full suite:
      `cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-test && ctest --test-dir build-test --output-on-failure`
- [ ] New focused tests cover: each of `index|load|unload` parsed; a rejected
      value; missing `--kserve_endpoint`; `load`/`unload` without
      `--kserve_model_name`; index rendering in text and JSON; an empty index;
      the typed unsupported-operation error producing the extension message, and
      any other client error *not* producing it.
- [ ] Build variants configure and build:
      - [ ] KServe off: `-DNEURIPLO_INFER_ENABLE_KSERVE=OFF` (and the flag then
            fails with the built-without-KServe message)
      - [ ] gRPC off: `-DNEURIPLO_INFER_ENABLE_GRPC=OFF` (HTTP path still works)
      - [ ] `-DWERROR=ON`
- [ ] `pre-commit run --all-files` clean.
- [ ] `python3 scripts/sync_supported_model_types.py --check` (no model types
      moved, so this must stay silent).
- [ ] `ctest --test-dir build-test -R capabilities_schema_contract` passes with
      the new flag advertised.
- [ ] `bash app/test/kserve_integration.sh` passes, including the new live `index`
      round trip.

## Manual

Against a local `neuriplo-kserve-runtime` with the Model Repository extension
enabled — record the exact commands and observed output:

- [ ] `--kserve_repository=index --kserve_endpoint=http://localhost:8000` lists the
      served models with name, version, state, and reason.
- [ ] The same with `--output_format=json` emits valid JSON (pipe through
      `python3 -m json.tool`).
- [ ] The same with `--kserve_transport=grpc` produces equivalent output.
- [ ] `--kserve_repository=unload --kserve_model_name=<m>` then `index` shows the
      model gone or `UNAVAILABLE`; `load` then `index` shows it back.
- [ ] No `--type`, `--weights`, or `--source` is required for any of the three.
- [ ] Against a server *without* the extension (Triton without
      `--model-control-mode=explicit`): the failure names the operation, the
      endpoint, and the server flag.
- [ ] `load` / `unload` of a model the server does not have: reports a missing
      model, with no extension hint attached.
- [ ] Against an unreachable endpoint: the existing transport error still appears,
      unchanged and unswallowed.
- [ ] Every doc link added resolves (`ls` for relative, `curl -sI` for absolute).

## Definition of Done

- [ ] Every requirement is implemented or explicitly deferred in
      [`requirements.md`](requirements.md).
- [ ] Nothing in *Out of Scope* was implemented — in particular no readiness
      probes, no `neuriplo-kserve-client` change, no load-then-wait polling.
- [ ] Nothing in `requirements.md` is left open, and no client gap was worked
      around in this repo.
- [ ] The `README.md` *Known Limitations* bullet about model management is gone.
- [ ] `CHANGELOG.md` `[Unreleased]` describes the flag and the client pin move;
      Phase 4 in [`../roadmap.md`](../roadmap.md) is `done`; the deferred readiness probes
      are recorded as a candidate.
- [ ] Spec, code, changelog, and roadmap tell the same story in one branch.

## Deviations

_Record anything run differently from the above, and why. Empty is a claim that
every box was executed as written._
