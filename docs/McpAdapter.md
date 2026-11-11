# MCP Adapter

Stdio Model Context Protocol adapter that exposes
[neuriplo-infer](https://github.com/olibartfast/neuriplo-infer) to MCP hosts
such as Hermes Agent and OpenCode. Implemented in
`scripts/mcp_neuriplo_adapter.py` (Python 3.12 standard library only, no new
dependency). Tests: `scripts/test_mcp_neuriplo_adapter.py`.

## What it exposes

| Tool | What it does |
|---|---|
| `neuriplo_capabilities` | Runs `neuriplo-infer --capabilities` and returns the JSON document (workflows, tasks, models, parameters, diagnostics). Takes no arguments. |
| `neuriplo_run` | Runs one local still-image inference in a fresh per-run directory and returns `exit_code`, `run_dir`, the parsed `run_report.json`, the rendered artifact path, and bounded stdout/stderr tails. |

MVP scope: single still-image local inference only. Remote KServe parameters,
multi-source/video inputs, and warmup/benchmark flags are rejected with an
explicit error, as are unknown arguments and wrong-typed fields (every
rejection names the flag and launches no process). Structured detection JSON
is not returned: the serialized result shape in the platform result contract
is still Draft with no producer, so the run report and the rendered image are
the machine-readable outputs.
See the [result contract](https://github.com/olibartfast/neuriplo-platform/blob/main/contracts/result-contract.md)
in [neuriplo-platform](https://github.com/olibartfast/neuriplo-platform) for the gap.

Freshness and isolation: each run executes in a unique run directory created
per call with `tempfile.mkdtemp` (under the system temp area, or under the
caller-given `workdir`), used as the child working directory, so the report
and artifact paths in a response always belong to that run. Proven isolation
is uniqueness-by-construction plus sequential request processing (calls are
served one at a time in arrival order over stdio); overlapping execution was
not exercised and is not claimed. The adapter never deletes run directories;
copy out needed artifacts. Bounded capture proven is retained buffers capped
by construction (`TAIL_BYTES` 256 KiB per stream on the run path via
`TailBuffer`; `CAPABILITIES_MAX_BYTES` 1 MiB on the capabilities path with
oversize rejected) with returned tails capped at `TAIL_CHARS` 4000 chars; no
peak-allocation measurement was made. Paths given to `neuriplo_run` are
resolved against the adapter process directory and frozen as absolute before
spawning. The binary is selected once at server startup (`--infer-bin`,
`NEURIPLO_INFER_BIN`, or `PATH`); there is no call-level override.

## Protocol

Legacy stdio handshake only: `initialize` (revisions up to `2025-11-25`),
`notifications/initialized`, `ping`, `tools/list`, `tools/call`. The
2026-07-28 revision (`server/discover`) is answered as unserved; both
supported hosts default to the legacy handshake. Malformed input yields a
parse error without killing the server; a missing id is treated as a
notification (no response) while a present null id is rejected as invalid
(the served 2025-11-25 revision forbids null request ids); unknown methods
yield Method Not Found. On timeout (and on readers lingering after normal
exit) the adapter terminates the child's process group, so descendants
holding the pipes cannot keep pump threads alive.

## Run it

```bash
# capabilities use whatever binary is configured
NEURIPLO_INFER_BIN=/path/to/neuriplo-infer \
  python3 scripts/mcp_neuriplo_adapter.py

# per-run timeout (seconds, default 300, max 3600)
NEURIPLO_MCP_TIMEOUT_S=600 \
NEURIPLO_INFER_BIN=/path/to/neuriplo-infer \
  python3 scripts/mcp_neuriplo_adapter.py
```

Set `NEURIPLO_INFER_BIN` to the `neuriplo-infer` build to expose (for example
`build-accel-test/app/neuriplo-infer`). Inference runs can take minutes: give
the host an execution timeout of at least 600 s (see configs below).

## Hermes Agent

From the [MCP config reference](https://hermes-agent.nousresearch.com/docs/reference/mcp-config-reference)
(fetched 2026-11-10): stdio servers use `command` plus `args`/`env` under
`mcp_servers`. Tool calls in Hermes time out per `timeout` (default 300 s);
raise it for inference.

```yaml
mcp_servers:
  neuriplo:
    command: "python3"
    args: ["/path/to/neuriplo-infer/scripts/mcp_neuriplo_adapter.py"]
    env:
      NEURIPLO_INFER_BIN: "/path/to/neuriplo-infer/build/app/neuriplo-infer"
      NEURIPLO_MCP_TIMEOUT_S: "600"
    timeout: 660
    tools:
      resources: false
      prompts: false
```

Reload servers after editing config with `/reload-mcp`. Hermes names the
tools `mcp__neuriplo__neuriplo_capabilities` and
`mcp__neuriplo__neuriplo_run`.

## OpenCode

From [MCP servers](https://opencode.ai/docs/mcp-servers/)
(fetched 2026-11-10 via https://opencode.ai/v2/docs/mcp-servers/, which
redirects to the canonical URL above): local servers use `mcp.servers` with `type: "local"`
and `command`. Execution timeouts default to 12 h; the catalog timeout
(`catalog`, default 30 s) covers tool listing, which is instant here.

```jsonc
// opencode.jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "neuriplo": {
        "type": "local",
        "command": ["python3", "/path/to/neuriplo-infer/scripts/mcp_neuriplo_adapter.py"],
        "environment": {
          "NEURIPLO_INFER_BIN": "/path/to/neuriplo-infer/build/app/neuriplo-infer",
          "NEURIPLO_MCP_TIMEOUT_S": "600",
        },
      },
    },
  },
}
```

Check the connection with `opencode mcp list` (expect `neuriplo connected`).
In this environment a project-local `opencode.json` was honored while the
`opencode.jsonc` file alone was not picked up. OpenCode names the tools
`neuriplo_neuriplo_capabilities` and `neuriplo_neuriplo_run`.

## Agent LLM supply (not this adapter)

To supply the agent's own LLM, point the agent at an external OpenAI-compatible server (vLLM or
`llama-server`), per the platform decision in
[ADR 0006](https://github.com/olibartfast/neuriplo-platform/blob/main/docs/adr/0006-generative-serving-over-openai-protocol.md).
That is a deployment choice, not an adapter feature.

## Tested versus proposed

- Unit (fake binary, no GPU): 36 tests in
  `scripts/test_mcp_neuriplo_adapter.py` covering stdio framing and envelope
  behavior, strict schema validation with zero process spawns on rejection,
  per-run isolation as uniqueness-by-construction plus sequential processing
  (a prior success is never attributed to a later failure; sequential calls
  return disjoint run dirs; overlapping execution not exercised or claimed),
  absolute-path handling across directories, returned-tail bound on noisy
  children (5 MB child, returned tails within 4000 chars, retained buffers
  capped by construction as above), timeout kill with the server left alive,
  and the load-bearing `--key=value` flag convention the owning repo's scripts
  and app tests use. Suite is warning-clean (`-W error::ResourceWarning`).
- Real binary (2026-11-10, `build-accel-test/app/neuriplo-infer`, backend
  `nert`): the capabilities probe through the adapter returns schema 2,
  workflows `local` + `client_server`, 11 tasks, server 0.2.1. A `neuriplo_run`
  with the empty `model.onnx` fails cleanly at `model_load` with exit 1, a
  fresh run directory, a parsed failed run report, no artifact, and the
  binary's own `empty model file` error. Weighted SUCCESS is blocked: the
  vendored `model.onnx` is 0 bytes and no CV-task ONNX weights exist anywhere
  in the workspace (only transformer node-test models in an unrelated uv cache
  and RF-DETR `.pt`/`.pth` checkpoints unusable with the NERT ONNX backend);
  no downloads were performed.
- OpenCode v2.2.6 (sandbox project-local `opencode.json` in scratch space;
  real user configs verified untouched): `opencode mcp list` reports `neuriplo
  connected`, i.e. the host's own initialize plus catalog against the
  adapter succeeded. Discovery only; per-host TOOL INVOCATION was
  attempted but not achieved: three `opencode run` sessions against the local
  Ollama `qwen2.5-coder:7b` (the only provider available without external
  credentials) showed the host presenting the tools (the model named
  `neuriplo_capabilities` in its output) but the model emitting text
  pseudo-calls instead of invoking - no `tool_result` event, and one reply
  hallucinated `schema_version 1.0` (actual: 2). No external provider session
  was available; no credentials configured or touched.
- Hermes v0.23.0 (scratch `HOME`; real `~/.hermes` verified untouched):
  `hermes mcp test neuriplo` reports connected in 523 ms with 2 tools
  discovered, both descriptions rendered. Discovery only; per-host TOOL
  INVOCATION was not exercised (needs a provider-backed agent session; no
  credentials configured or touched).
- Proposed, not attested: live weighted inference end to end; Streamable HTTP
  transport; structured detection output (blocked on the result-contract
  Draft noted above).
