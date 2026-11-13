# Feature Requirements — MCP adapter for agent hosts

Roadmap phase: standalone tooling (outside phased delivery, like the
LightGlue/NERT entries); retroactive packet for already-built work.
Branch: `feature/mcp-agent-tools`

## Goal

Let MCP-capable agent hosts (Hermes Agent, OpenCode) query
neuriplo-infer capabilities and run single still-image local inferences
through the Model Context Protocol over stdio, without adding any runtime
dependency beyond the Python 3 standard library. This branch is already
implemented (PR #111); this packet records what was built, not what is
planned.

## In Scope

- `scripts/mcp_neuriplo_adapter.py`: stdio MCP server (stdlib only)
  exposing exactly two tools:
  - `neuriplo_capabilities` — runs `neuriplo-infer --capabilities`,
    returns the JSON document; takes no arguments.
  - `neuriplo_run` — runs one local still-image inference in a fresh
    per-run directory; returns `exit_code`, `run_dir`, the parsed
    `run_report.json`, the rendered artifact path, and bounded
    stdout/stderr tails.
- Legacy stdio handshake only (`initialize` up to revision `2025-11-25`,
  `notifications/initialized`, `ping`, `tools/list`, `tools/call`);
  revision `2026-07-28` answered as unserved. Malformed input yields a
  parse error without killing the server.
- Strict argument validation: remote KServe parameters,
  multi-source/video inputs, warmup/benchmark flags, unknown arguments,
  and wrong-typed fields are rejected with an error naming the flag, and
  no process is spawned.
- Isolation by construction: unique run directory per call
  (`tempfile.mkdtemp`), used as the child working directory; sequential
  request processing. Bounded capture (`TAIL_BYTES` 256 KiB per stream
  retained, `TAIL_CHARS` 4000 chars returned). Process-group kill on
  timeout. Binary selected once at startup (`--infer-bin`,
  `NEURIPLO_INFER_BIN`, or `PATH`).
- `scripts/test_mcp_neuriplo_adapter.py`: 40-test stdio suite (fake
  binary, no GPU), warning-clean under `-W error::ResourceWarning`.
- `docs/McpAdapter.md`: host configuration for Hermes Agent and
  OpenCode, plus tested-versus-proposed evidence, including the weighted
  RF-DETR Nano success run through the adapter.
- `CHANGELOG.md` `[Unreleased]` entries for the adapter hardening and
  the weighted RF-DETR Nano attestation.

## Out of Scope

- Host-driven (model-driven) tool invocation: OpenCode discovery
  (`opencode mcp list` → `neuriplo connected`) and Hermes discovery
  (`hermes mcp test neuriplo` → 2 tools) are attested; per-host TOOL
  INVOCATION was attempted but not achieved and remains unattested.
- Streamable HTTP transport: proposed in `docs/McpAdapter.md`, not built.
- Structured detection JSON output: blocked on the platform result
  contract, still Draft with no producer; the run report and rendered
  image are the machine-readable outputs.

## Decisions

- Stdlib-only Python adapter instead of an MCP SDK: zero new
  dependencies, per [`../tech-stack.md`](../tech-stack.md) minimalism;
  the legacy handshake the supported hosts default to is small enough to
  implement directly.
- Thin process wrapper, not an inference implementation: the adapter
  shells out to an existing `neuriplo-infer` binary with an argument
  array (never a shell string) and returns surfaces that binary already
  produces, so CLI compatibility and output schema are untouched.
- No call-level binary override: the binary is frozen at server startup
  so a tool call cannot redirect execution to an arbitrary binary.
- Host configs live in `docs/McpAdapter.md`, not in repo dotfiles: host
  wiring is a deployment choice (like the agent-LLM supply per platform
  ADR 0006), and real user configs were verified untouched during testing.

## Constraints and Context

- Constitution rules: preserve CLI compatibility and output schema;
  no inference-semantics change (mechanical wiring only); keep the change
  small and reviewable.
- No sibling-repo change or pin move required: the adapter consumes the
  already-built binary and the pinned neuriplo-tasks model types.
- Contract surfaces touched: none in the app (`--capabilities` output,
  `docs/capabilities.schema.json`, and generated model-type docs are
  unchanged by this branch).

## Open Questions

- None blocking this branch. The known follow-ups (host-driven
  invocation attestation with an external provider, HTTP transport,
  structured detection JSON once the result contract lands) are recorded
  as Out of Scope / proposed, not as open questions on this merge.
