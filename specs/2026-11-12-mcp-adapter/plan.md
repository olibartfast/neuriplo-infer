# Feature Plan — MCP adapter for agent hosts

> Retroactive packet for PR #111 (`feature/mcp-agent-tools`), written after
> implementation to record what was built. Derived from `requirements.md`.

## Group 1 — Foundation

1. Implement `scripts/mcp_neuriplo_adapter.py`: stdio framing, legacy
   handshake (`initialize`, `notifications/initialized`, `ping`,
   `tools/list`, `tools/call`), strict `neuriplo_run` argument schema
   with reject-before-spawn, per-run directory isolation, bounded
   tail capture, and process-group cleanup on timeout — stdlib only.

## Group 2 — Behavior

2. Wire the two tools to the binary: `neuriplo_capabilities`
   (`--capabilities` JSON passthrough) and `neuriplo_run` (one local
   still-image inference in a fresh per-call directory, returning
   `exit_code`, `run_dir`, parsed `run_report.json`, rendered artifact
   path, bounded tails).
3. Fix argument validation and JSON-RPC parsing gaps found during review
   (still-image allowlist, optional argument types, oversized integer ids).

## Group 3 — Contract and documentation

4. Add `docs/McpAdapter.md` with Hermes Agent and OpenCode configs and
   tested-versus-proposed evidence, including the weighted RF-DETR Nano
   success run. Add `CHANGELOG.md` `[Unreleased]` entries. No
   `--capabilities`, schema, or generated-docs surfaces move, so no
   regeneration was needed.

## Group 4 — Verification

5. Add `scripts/test_mcp_neuriplo_adapter.py` (40 tests: framing and
   envelope behavior, zero-spawn rejection, isolation, absolute-path
   handling, tail bounds, timeout kill, `--key=value` convention).
6. Run the checks declared in `validation.md` and record honest results
   there.
