# Feature Plan — KServe model repository through the CLI

Derived from [`requirements.md`](requirements.md). Commit by group; each group
ends in something observable.

Do not start Group 1 until Phase 3's typed error is released upstream and
`NEURIPLO_KSERVE_CLIENT_VERSION` in `versions.env` points at it.

## Group 1 — Client construction

1. Move the `versions.env` pin to the client release carrying the typed error and
   confirm the existing KServe tests pass on it before writing new code.
   Clients are constructed with an empty model name — settled by reading the
   client, see *Resolved Questions* in `requirements.md`. No placeholder.
2. Extract the transport-selecting client construction duplicated in
   `app/src/InferencePipeline.cpp` (`fetchTaskModelMetadata`, ~line 109, and the
   builder path, ~line 233) into one factory taking endpoint, model, version, and
   timeout. Both existing call sites move onto it; behavior is unchanged.
3. Observable: the test build passes with the factory in place and no behavioral
   diff — the existing KServe tests still pass untouched.

## Group 2 — Flag, validation, and dispatch

4. Declare `--kserve_repository` in the `params` string in
   `app/src/CommandLineParser.cpp`, and add `kserve_repository` to `AppConfig`.
5. Validate: the value is one of `index|load|unload`; `--kserve_endpoint` is
   required; `load` and `unload` require `--kserve_model_name`. Each failure names
   the flag and the fix, in the style of the existing parser errors.
6. Short-circuit in the parser the way `capabilities` does, so no `--type`,
   `--weights`, or `--source` is demanded, and dispatch from `app/main.cpp`
   before `NeuriploInfer` is constructed.
7. Under a local-only build (`NEURIPLO_INFER_ENABLE_KSERVE=OFF`), passing the flag
   fails with a message saying the binary was built without KServe support.
8. Observable: `--kserve_repository=index --kserve_endpoint=...` reaches the
   client and prints something; the three validation failures print their errors.

## Group 3 — Operations and output

9. Implement the three operations behind one command type, alongside the existing
   commands in `app/inc/CLICommands.hpp` / `app/src/CLICommands.cpp`. It takes a
   client rather than an `InferencePipeline` — do not force it into the pipeline
   command interface if that costs a fake pipeline.
10. Render `index` as an aligned text table (name, version, state, reason) and as
    a JSON array under `--output_format=json`. Empty index is a valid result, not
    an error, and says so in text mode.
11. `load` / `unload`: confirmation line naming model and endpoint; non-zero exit
    on failure.
12. Classify the failure with Phase 3's typed error. An unsupported operation
    (HTTP 4xx / gRPC `UNIMPLEMENTED`) reports the operation, the endpoint, and
    the server-side setting (Triton: `--model-control-mode=explicit`). Every other
    failure — missing model, transport, TLS — keeps its own message and gains no
    misleading hint. Never parse the message text.
13. A client built with the strict `OIP` proto profile throws "model repository
    gRPC not compiled in" from the gRPC path. Surface that as a build-side
    problem naming `KSERVE_CLIENT_PROTO_PROFILE`, not as a server-side one, and
    point at `--kserve_transport=http` as the way through.

## Group 4 — Contract and documentation

14. Add the flag and its enum values to `app/src/Capabilities.cpp`; confirm
    `docs/capabilities.schema.json` still accepts the output, and extend the
    schema only if it constrains the flag set.
15. `docs/KserveRuntime.md`: a section covering the three operations, both
    transports, the extension requirement, and the proto-profile caveat.
16. `README.md`: drop the *Known Limitations* bullet stating these are not
    exposed; add the flag to the option list and one usage example.
    `CHANGELOG.md`: an `### Added` entry under `[Unreleased]`.
17. Mark Phase 4 `done` in [`../roadmap.md`](../roadmap.md); add the deferred
    readiness probes as a roadmap candidate.

## Group 5 — Verification

18. Unit tests: flag parsing and each validation failure
    (`app/test/test_parseCommandLineArguments.cpp`); index rendering in both
    output formats including the empty index; a fake client throwing the typed
    unsupported-operation error produces the extension message, and one throwing
    any other error does *not* (`app/test/test_KserveEngine.cpp` shows the
    fake-client pattern).
19. Extend `app/test/kserve_integration.sh` with a live `index` round trip.
20. Run everything declared in [`validation.md`](validation.md).
