# Feature Requirements — KServe model repository through the CLI

Roadmap phase: [Phase 4 — Expose KServe model management through the CLI](../roadmap.md#phase-4--expose-kserve-model-management-through-the-cli--not-started)
Branch: `feature/kserve-repository-cli`
Blocked on: [Phase 3 — Typed errors in `neuriplo-kserve-client`](../roadmap.md#phase-3--typed-errors-in-neuriplo-kserve-client--not-started)

## Goal

Make the KServe V2 Model Repository extension reachable from the command line.
`kserve::IClient::repositoryIndex()`, `loadModel()`, and `unloadModel()` are
implemented on both the HTTP and gRPC clients in `neuriplo-kserve-client`, but no
flag reaches them, so a user running against `neuriplo-kserve-runtime` or Triton
has to leave this app to see what the server holds or to load a model before
inferring against it.

## In Scope

- A new action flag `--kserve_repository=<index|load|unload>` that runs instead of
  inference and exits, in the manner of `--capabilities` and `--export_metadata`.
  It requires no `--source`, no `--weights`, and no `--type`.
- `index`: prints every `kserve::RepositoryModel` the server reports — name,
  version, state, and reason — honouring `--output_format=text|json`.
- `load` / `unload`: act on `--kserve_model_name`, print a confirmation naming
  the model and the server, and exit non-zero on failure.
- Both transports: `--kserve_transport=http|grpc` selects the client exactly as it
  does for inference, and `--kserve_endpoint`, `--kserve_timeout_ms`, and the
  existing TLS environment (`KSERVE_CLIENT_CERT` / `KSERVE_CLIENT_KEY`) apply
  unchanged.
- A server that does not enable the extension produces an error naming the
  endpoint, the operation, and what the server needs (for Triton,
  `--model-control-mode=explicit`) — not a bare HTTP status or gRPC code. The
  unsupported case is identified from the typed error delivered by Phase 3, not
  from the text of the message.
- `--capabilities` advertises the new flag and its enum values, and the output
  still validates against `docs/capabilities.schema.json`.
- `docs/KserveRuntime.md` documents the three operations; the *Known Limitations*
  entry in `README.md` that says they are not exposed is removed.

## Out of Scope

- Liveness, readiness, and model-ready probes as CLI actions. They are a separate
  surface and were considered and deferred; add a roadmap candidate rather than
  smuggling them in here.
- Any change to `neuriplo-kserve-client` *on this branch*. The typed error this
  feature needs is Phase 3's work, landing upstream and arriving here as a pin
  move in `versions.env` before this branch starts (see
  [`../tech-stack.md`](../tech-stack.md#explicit-non-choices)). If a further gap
  appears mid-implementation, stop and raise it there — do not work around it here.
- Repository operations against local backends. This flag is meaningless without
  `--kserve_endpoint` and must say so.
- Polling or waiting for a model to reach `READY` after `load`. The call returns
  when the server accepts it; a user who wants readiness runs `index` again.

## Decisions

- **An action flag, not a subcommand.** `cv::CommandLineParser` has no positional
  subcommand concept today, and `--capabilities` / `--export_metadata` already
  establish the "action flag that runs instead of inference" idiom. A subcommand
  would be a larger CLI contract change than the feature warrants.
- **A single enum flag rather than three booleans.** `--kserve_repository=load`
  makes the operations mutually exclusive by construction; three boolean flags
  would need pairwise validation and a fourth error message.
- **Attempt the call, then translate the failure.** Probing `/v2` for the
  advertised extension list costs an extra round trip and trusts servers to
  report extensions honestly; several do not. Issue the call and map a 4xx or
  `UNIMPLEMENTED` onto an actionable message instead.
- **Short-circuit before the pipeline, like `--capabilities`.** Repository
  operations need a client, not an `InferencePipeline`: no weights, no task, no
  renderer. Building the pipeline first would demand a `--type` the operation has
  no use for. This is the one structural difference from `--export_metadata`,
  which does go through the full builder.
- **Reuse `--kserve_model_name` for load/unload** rather than adding a
  repository-specific name flag: it already means "the model on the server".
- **Fix the error classification upstream first, rather than working around it
  here.** `neuriplo-kserve-client` v0.4.0 throws a plain `std::runtime_error` from
  both transports with the status flattened into the message
  (`KserveHttpClient.cpp:513`, `throwStatus` at `KserveGrpcClient.cpp:75`), so the
  app cannot tell an extension-disabled 4xx/`UNIMPLEMENTED` from a missing model
  without matching on upstream's error prose. The alternative considered — report
  the message verbatim and always append the extension hint — needs no upstream
  change, but attaches a hint to failures it does not explain and leaves every
  consumer of the client with the same gap. This is exactly the client gap
  `tech-stack.md` says to escalate, so it becomes
  [Phase 3](../roadmap.md#phase-3--typed-errors-in-neuriplo-kserve-client--not-started)
  and this feature consumes the result.

## Constraints and Context

- Constitution: no new third-party dependency; no silent CLI breakage (this is
  purely additive); a capability may not be advertised before it is routed
  ([`../tech-stack.md`](../tech-stack.md)).
- Existing patterns to reuse:
  - `app/src/CommandLineParser.cpp` — the `capabilities` early-return at ~line 134
    and the flag-declaration string at the top of the file.
  - `app/main.cpp` — where `show_capabilities` is dispatched before
    `NeuriploInfer` is constructed.
  - `fetchTaskModelMetadata` in `app/src/InferencePipeline.cpp:109` — the existing
    transport-selecting client construction, duplicated once more at ~line 233.
    A third copy is the wrong answer; extract a small factory.
  - `app/src/Capabilities.cpp` — how an enum flag is described.
- Build variants: the flag exists only under `NEURIPLO_INFER_ENABLE_KSERVE`. A
  local-only build must still configure, build, and give a clear error if the flag
  is passed. A build without gRPC must still serve the HTTP path.
- Contract surfaces touched: CLI flags, `--capabilities` output,
  `docs/capabilities.schema.json` (only if the schema constrains the flag list),
  `CHANGELOG.md`, `README.md`, `docs/KserveRuntime.md`.
- Cross-repo: Phase 3 lands the typed error in `neuriplo-kserve-client` and
  `NEURIPLO_KSERVE_CLIENT_VERSION` in `versions.env` moves off `v0.4.0` before
  this branch starts. The repository *API* (`repositoryIndex` / `loadModel` /
  `unloadModel`) already exists at `v0.4.0`; only the error type is new.
- The client's gRPC repository RPCs are compiled behind
  `KSERVE_CLIENT_PROTO_REPOSITORY`, set by its `KSERVE_CLIENT_PROTO_PROFILE`
  option (`OIP_REPOSITORY` by default, which this repo does not override). Built
  with the strict `OIP` profile, all three gRPC operations throw "model
  repository gRPC not compiled in" at runtime
  (`KserveGrpcClient.cpp:450`). That message must reach the user intelligibly
  rather than being reported as a server-side problem.

## Resolved Questions

Both were closed by reading `neuriplo-kserve-client` v0.4.0 as pinned in
`versions.env`; neither needed a live server.

- **Constructing a client with an empty model name is safe for all three
  operations.** Neither constructor validates the name
  (`KserveHttpClient.cpp:441`, `KserveGrpcClient.cpp:214` — the gRPC one uses only
  the endpoint, for the channel). The HTTP repository calls go through
  `postRepository` with paths from `repositoryIndexPath()` (a constant
  `/v2/repository/index`) and `repositoryModelLoadPath(model_name)` / its unload
  twin, which take the *argument*, not the member. The gRPC calls leave
  `RepositoryIndexRequest` default and set the name on load/unload from the
  argument. `model_name_` is read only by `modelMetadata`, `modelReady`, and
  `infer`, none of which this feature calls. No placeholder name, and no upstream
  request.

- **No guard on unloading the model named by `--kserve_model_name`.** The flag
  short-circuits in the parser before any pipeline is built, so no inference can
  run in the same invocation — there is nothing for the guard to protect. The
  question was moot, as suspected.
