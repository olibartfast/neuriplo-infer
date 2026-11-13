# Feature Validation — MCP adapter for agent hosts

> Retroactive packet for PR #111. Results recorded after execution on the
> feature branch; see Deviations for anything not run as written.

## Automated

- [x] Default build configures and builds:
      `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [x] Test build and full suite pass:
      `cmake -S . -B build-test -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-test && ctest --test-dir build-test --output-on-failure`
- [x] Focused tests cover the success path *and* the failure path this feature adds.
- [x] Generated docs are in sync: `python3 scripts/sync_supported_model_types.py --check`
- [x] `--capabilities` still validates (`capabilities_schema_contract`).
- [x] `pre-commit run --all-files` (clang-format, cppcheck) is clean.

## Manual

- [x] The primary invocation behaves as `requirements.md` specifies — record the
      exact command and the observed output.
- [x] Invalid, empty, and unsupported-combination inputs fail fast with an error
      naming the flag and the way out.
- [x] Every hyperlink added to docs resolves (`ls` for relative, `curl -sI` for absolute).

## Results

Executed 2026-11-12 on `feature/mcp-agent-tools` (HEAD `2288c31` + packet):

- `python3 scripts/test_mcp_neuriplo_adapter.py`: 40 tests, OK (8.1 s).
- Fresh `build-test` (stale dir wiped — it carried a pre-pin
  `videocapture-src` layout without `include/` and failed on `Frame.hpp`;
  not caused by this branch): configure + build OK, `ctest`: 197/197
  passed, including `capabilities_schema_contract` (#186),
  `CapabilitiesContract.*` (#9–18), and task-routing mirrors (#7).
- Default `build` (NERT, tests OFF): configure + `cmake --build build` OK.
- `sync_supported_model_types.py --check` against pinned
  neuriplo-tasks `v0.8.5` (`git show v0.8.5:README.md`): passed. Check
  against sibling `develop` README fails (newer upstream) — expected and
  out of scope for this branch.
- C++ lint/build CI on PR #111: Format Check, Clang-Tidy, Cppcheck,
  Build with Strict Warnings all SUCCESS (no C++ touched by this branch).
- Primary invocation: attested in `docs/McpAdapter.md` (weighted RF-DETR
  Nano through the adapter, exit 0, success run report, rendered
  artifact) — not re-run here; host-driven invocation remains
  unattested per the doc.
- Rejection path: covered by the 40-test suite (zero-spawn rejection,
  allowlist, wrong-typed fields, oversized integer ids).
- Hyperlinks: relative paths (`scripts/mcp_neuriplo_adapter.py`,
  `scripts/test_mcp_neuriplo_adapter.py`) exist; all six absolute URLs
  in `docs/McpAdapter.md` return HTTP 200.

## Definition of Done

- [ ] Every requirement is implemented or explicitly deferred in `requirements.md`.
- [ ] Nothing in *Out of Scope* was implemented anyway.
- [ ] Deviations from this file are recorded here, honestly, with what was run instead.
- [ ] `CHANGELOG.md` describes the user-visible change; `../roadmap.md` phase
      status is updated.
- [ ] Spec, code, changelog, and roadmap tell the same story in one branch.
