#!/usr/bin/env python3
"""Minimal stdio MCP adapter that exposes neuriplo-infer to agents.

This is a thin process wrapper, not an inference implementation. It shells out
to an existing neuriplo-infer binary with an argument array (never a shell
string) and returns the surfaces that binary already produces:

- neuriplo_capabilities: neuriplo-infer --capabilities (JSON to stdout)
- neuriplo_run: one local still-image inference in a fresh per-run directory;
  returns exit_code, run_dir, the parsed run_report.json, the rendered
  artifact path, and bounded stdout/stderr tails.

MVP scope: single still-image local inference only. Remote KServe parameters,
multi-source/video inputs, and warmup/benchmark flags are rejected with an
explicit error. Structured detection JSON is not returned: the serialized
result shape in the platform result contract is still Draft with no producer,
so the run report and the rendered image are the machine-readable outputs.

Freshness and isolation: every run executes in a unique run directory created
per call (under the system temp area, or under the caller-given workdir when
one is supplied), used as the child working directory. The fixed
data/output/run_report.json path inside that directory is therefore always
fresh, and each run's outputs are unique by construction. Requests are served
one at a time in arrival order; overlapping execution was not exercised and
is not claimed. The adapter never deletes run directories; the caller copies
out needed artifacts.

Path semantics: source/weights/labels are validated against the adapter
process directory and normalized to absolute paths before spawning, so their
meaning does not change with the child working directory. The binary is
resolved once at server startup (flag, env, or PATH) and absolutized there;
there is no call-level binary override.

Protocol: newline-delimited JSON-RPC 2.0 over stdio with the legacy
initialize handshake (protocol revisions up to 2025-11-25). The 2026-07-28
revision (server/discover) is not served; both supported hosts default to the
legacy handshake. Logging goes to stderr; stdout carries protocol messages
only. Child output is pumped through buffers that retain at most TAIL_BYTES
per stream (256 KiB), and returned tails are capped at TAIL_CHARS characters,
so a noisy child stays bounded in what the adapter retains and returns; no
peak-allocation measurement is made. On timeout (and on readers lingering
after normal exit) the adapter terminates the child's process group, so
descendants holding the pipes cannot keep pump threads alive; partial tails
are decoded safely.

Environment:
  NEURIPLO_INFER_BIN  path to the neuriplo-infer binary (default: PATH lookup)
  NEURIPLO_MCP_TIMEOUT_S  default per-run timeout in seconds (default: 300)
  NEURIPLO_MCP_CAPABILITIES_TIMEOUT_S  capabilities timeout in seconds (default: 60)
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


ADAPTER_VERSION = "0.2.1"
SERVER_NAME = "neuriplo-mcp"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = "2025-11-25"
DEFAULT_TIMEOUT_S = 300
MAX_TIMEOUT_S = 3600
TAIL_BYTES = 262144  # per-stream bytes retained while the child runs
TAIL_CHARS = 4000  # characters returned per stream
READ_CHUNK = 65536
THREAD_JOIN_S = 10.0
LINGER_JOIN_S = 2.0  # post-exit grace before treating readers as stuck
RUN_DIR_PREFIX = "neuriplo-mcp-"
CAPABILITIES_MAX_BYTES = 1048576
DEFAULT_CAPABILITIES_TIMEOUT_S = 60

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

RUN_ARG_KEYS = frozenset({
    "type", "source", "weights", "labels", "min_confidence", "nms_threshold",
    "use_gpu", "timeout_s", "workdir",
})

OUT_OF_SCOPE_KEYS = frozenset({
    "kserve_endpoint", "kserve_model_name", "kserve_model_version",
    "kserve_transport", "kserve_timeout_ms", "input_mode", "task_model",
    "task_model_version", "postprocess_mode", "warmup", "benchmark",
    "iterations", "output_video", "timings_csv", "sources",
})


def log(message):
    sys.stderr.write("[%s] %s\n" % (SERVER_NAME, message))
    sys.stderr.flush()


def resolve_infer_bin(explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    configured = os.environ.get("NEURIPLO_INFER_BIN")
    if configured:
        return os.path.abspath(configured)
    found = shutil.which("neuriplo-infer")
    return found or "neuriplo-infer"


def default_timeout():
    try:
        value = int(os.environ.get("NEURIPLO_MCP_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))
    except ValueError:
        return DEFAULT_TIMEOUT_S
    if value < 1 or value > MAX_TIMEOUT_S:
        return DEFAULT_TIMEOUT_S
    return value


def capabilities_timeout():
    try:
        value = int(os.environ.get("NEURIPLO_MCP_CAPABILITIES_TIMEOUT_S",
                                   str(DEFAULT_CAPABILITIES_TIMEOUT_S)))
    except ValueError:
        return DEFAULT_CAPABILITIES_TIMEOUT_S
    if value < 1 or value > MAX_TIMEOUT_S:
        return DEFAULT_CAPABILITIES_TIMEOUT_S
    return value


def tail(text, limit=TAIL_CHARS):
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def error_response(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def tool_text_result(payload, is_error=False):
    result = {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}
    if is_error:
        result["isError"] = True
    return result


def negotiate_protocol(client_version):
    if client_version in SUPPORTED_PROTOCOL_VERSIONS:
        return client_version
    return DEFAULT_PROTOCOL_VERSION


def capabilities_tool_schema():
    return {
        "name": "neuriplo_capabilities",
        "description": (
            "Return the machine-readable capabilities of the configured "
            "neuriplo-infer binary (workflows, tasks, models, parameters, "
            "diagnostics). Read-only; takes no arguments."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    }


def run_tool_schema():
    return {
        "name": "neuriplo_run",
        "description": (
            "Run one local still-image inference with neuriplo-infer in a "
            "fresh per-run directory and return exit_code, run_dir, the "
            "parsed run report, the rendered artifact path, and bounded "
            "stdout/stderr tails. MVP scope only: single image, local "
            "backends. Remote KServe, video, warmup, and benchmark inputs are "
            "rejected. Structured detections are not returned (result "
            "contract serialized shape is Draft with no producer)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "Model type string (e.g. yolo)"},
                "source": {"type": "string", "description": "Path to one still-image file (resolved against the adapter directory, then frozen as absolute)"},
                "weights": {"type": "string", "description": "Path to model weights file or directory"},
                "labels": {"type": "string", "description": "Optional path to class labels file"},
                "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "nms_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "use_gpu": {"type": "boolean", "description": "Pass --use-gpu to the binary (must be a real boolean)"},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_S},
                "workdir": {"type": "string", "description": "Existing directory under which the per-run directory is created (default: system temp)"},
            },
            "required": ["type", "source", "weights"],
            "additionalProperties": False,
        },
    }


def error_data(flag, message):
    return {"flag": flag, "message": message}


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_run_args(args):
    if not isinstance(args, dict):
        return None, error_data("arguments", "arguments must be an object")
    for key in sorted(set(args)):
        if key in OUT_OF_SCOPE_KEYS:
            return None, error_data(
                "--" + key,
                "out of MVP scope: the MCP adapter supports one local "
                "still-image run only; remote KServe, video, warmup, and "
                "benchmark inputs are rejected")
        if key not in RUN_ARG_KEYS:
            return None, error_data("--" + key, "unknown argument: " + key)
    for key in ("type", "source", "weights"):
        value = args.get(key)
        if not isinstance(value, str) or not value:
            return None, error_data("--" + key, "missing required string argument: " + key)
    labels = args.get("labels")
    if "labels" in args and not isinstance(labels, str):
        return None, error_data("--labels", "labels must be a path string")
    workdir = args.get("workdir")
    if "workdir" in args and not isinstance(workdir, str):
        return None, error_data("--workdir", "workdir must be a path string")
    use_gpu = args.get("use_gpu", False)
    if not isinstance(use_gpu, bool):
        return None, error_data("--use_gpu", "use_gpu must be a boolean")
    for key in ("min_confidence", "nms_threshold"):
        value = args.get(key)
        if key in args:
            if not _is_number(value):
                return None, error_data("--" + key, "must be a number between 0 and 1")
            if not 0.0 <= float(value) <= 1.0:
                return None, error_data("--" + key, "must be between 0 and 1")
    timeout = args.get("timeout_s", default_timeout())
    if type(timeout) is not int:
        return None, error_data("--timeout_s", "timeout_s must be an integer number of seconds")
    if timeout < 1 or timeout > MAX_TIMEOUT_S:
        return None, error_data("--timeout_s", "timeout_s must be between 1 and %d" % MAX_TIMEOUT_S)
    if "," in args["source"]:
        return None, error_data(
            "--source",
            "MVP supports exactly one still-image source; multi-source input "
            "is out of scope (optical flow / keypoint matching stay on the CLI)")
    if Path(args["source"]).suffix.lower() not in IMAGE_SUFFIXES:
        return None, error_data(
            "--source", "MVP source must be a still image with one of these "
            "extensions: %s" % ", ".join(sorted(IMAGE_SUFFIXES)))
    source = os.path.abspath(args["source"])
    if not os.path.isfile(source):
        return None, error_data("--source", "source does not exist or is not a file: %s" % args["source"])
    weights = os.path.abspath(args["weights"])
    if not os.path.exists(weights):
        return None, error_data("--weights", "weights path does not exist: %s" % args["weights"])
    if labels is not None:
        labels = os.path.abspath(labels)
        if not os.path.isfile(labels):
            return None, error_data("--labels", "labels file does not exist: %s" % args["labels"])
    if workdir is not None and not os.path.isdir(workdir):
        return None, error_data("--workdir", "workdir does not exist: %s" % args["workdir"])
    return {
        "type": args["type"],
        "source": source,
        "weights": weights,
        "labels": labels,
        "min_confidence": args.get("min_confidence"),
        "nms_threshold": args.get("nms_threshold"),
        "use_gpu": use_gpu,
        "timeout_s": timeout,
        "workdir": workdir,
    }, None


def build_argv(validated, infer_bin):
    # Flag spelling follows the owning repo's own convention (see
    # docker_run_inference_example.sh and the app tests): valued flags use
    # --key=value form, booleans are bare. Space-separated valued flags
    # misparse in this binary's OpenCV CommandLineParser (source reads as
    # "true"), so the = form is load-bearing, not stylistic.
    argv = [infer_bin, "--type=%s" % validated["type"], "--source=%s" % validated["source"],
            "--weights=%s" % validated["weights"], "--no_display"]
    if validated["labels"]:
        argv += ["--labels=%s" % validated["labels"]]
    if validated["min_confidence"] is not None:
        argv += ["--min_confidence=%s" % validated["min_confidence"]]
    if validated["nms_threshold"] is not None:
        argv += ["--nms_threshold=%s" % validated["nms_threshold"]]
    if validated["use_gpu"]:
        argv += ["--use-gpu"]
    return argv


class TailBuffer:
    """Thread-safe byte buffer retaining only the newest cap bytes."""

    def __init__(self, cap=TAIL_BYTES):
        self.cap = cap
        self.lock = threading.Lock()
        self.data = bytearray()
        self.total = 0

    def feed(self, chunk):
        if not chunk:
            return
        with self.lock:
            self.total += len(chunk)
            self.data += chunk
            if len(self.data) > self.cap:
                del self.data[:len(self.data) - self.cap]

    def text(self):
        with self.lock:
            raw = bytes(self.data)
        return raw.decode("utf-8", errors="replace")

    def total_bytes(self):
        with self.lock:
            return self.total


def _drain(stream, buf):
    try:
        while True:
            chunk = stream.read(READ_CHUNK)
            if not chunk:
                break
            buf.feed(chunk)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def spawn_bounded(argv, rundir, timeout_s):
    """Run argv with cwd=rundir, pumping both streams with a memory cap.

    The child runs in its own process group so timeout cleanup terminates
    the whole tree: a descendant retaining the pipes cannot keep the pump
    threads (or itself) alive past the call.
    """
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=rundir, start_new_session=True)
    out_buf, err_buf = TailBuffer(), TailBuffer()
    readers = [threading.Thread(target=_drain, args=(proc.stdout, out_buf),
                                name="mcp-drain-stdout", daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, err_buf),
                                name="mcp-drain-stderr", daemon=True)]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        return_code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        return_code = proc.wait()
    lingering = _join_readers(readers, THREAD_JOIN_S if timed_out else LINGER_JOIN_S)
    if lingering:
        # Child is gone but something it spawned still holds the pipes.
        _kill_tree(proc)
        _join_readers(readers, THREAD_JOIN_S)
    return return_code, timed_out, out_buf.text(), err_buf.text()


def _kill_tree(proc):
    """Terminate the child's process group; fall back to the child alone."""
    killpg = getattr(os, "killpg", None)
    if killpg is not None:
        try:
            killpg(proc.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _join_readers(readers, timeout):
    """Join pump threads; return those still alive (pipes held by survivors)."""
    lingering = []
    for reader in readers:
        reader.join(timeout=timeout)
        if reader.is_alive():
            lingering.append(reader)
    return lingering


def read_report(rundir):
    report_path = Path(rundir) / "data" / "output" / "run_report.json"
    if not report_path.is_file():
        return None, "no run report at %s" % report_path
    try:
        return json.loads(report_path.read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, "could not parse %s: %s" % (report_path, exc)


def find_artifact(rundir):
    output_dir = Path(rundir) / "data" / "output"
    if not output_dir.is_dir():
        return None
    candidates = [p for p in output_dir.glob("processed_*") if p.is_file()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def run_capabilities(infer_bin):
    timeout_s = capabilities_timeout()
    try:
        proc = subprocess.Popen(
            [infer_bin, "--capabilities"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
    except FileNotFoundError:
        return None, "neuriplo-infer binary not found: %s" % infer_bin
    except OSError as exc:
        return None, "could not launch neuriplo-infer: %s" % exc
    out_buf = TailBuffer(cap=CAPABILITIES_MAX_BYTES)
    err_buf = TailBuffer()
    readers = [threading.Thread(target=_drain, args=(proc.stdout, out_buf),
                                name="mcp-drain-stdout", daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, err_buf),
                                name="mcp-drain-stderr", daemon=True)]
    for reader in readers:
        reader.start()
    return_code = None
    elapsed = 0.0
    slice_s = 0.2
    timed_out = False
    oversized = False
    while return_code is None:
        remaining = timeout_s - elapsed
        if remaining <= 0:
            timed_out = True
            break
        try:
            return_code = proc.wait(timeout=min(slice_s, remaining))
        except subprocess.TimeoutExpired:
            if out_buf.total_bytes() > CAPABILITIES_MAX_BYTES:
                oversized = True
                break
            elapsed += min(slice_s, remaining)
    if timed_out:
        _kill_tree(proc)
        try:
            proc.wait(timeout=THREAD_JOIN_S)
        except subprocess.TimeoutExpired:
            pass
        _join_readers(readers, THREAD_JOIN_S)
        return None, "neuriplo-infer --capabilities timed out"
    if oversized:
        _kill_tree(proc)
        proc.wait()
        _join_readers(readers, THREAD_JOIN_S)
        return None, ("neuriplo-infer --capabilities output exceeded "
                      "%d bytes limit (1 MiB)" % CAPABILITIES_MAX_BYTES)
    if _join_readers(readers, LINGER_JOIN_S):
        # Child is gone but something it spawned still holds the pipes.
        _kill_tree(proc)
        _join_readers(readers, THREAD_JOIN_S)
    if out_buf.total_bytes() > CAPABILITIES_MAX_BYTES:
        return None, ("neuriplo-infer --capabilities output exceeded "
                      "%d bytes limit (1 MiB)" % CAPABILITIES_MAX_BYTES)
    if return_code != 0:
        return None, "neuriplo-infer --capabilities exited %d: %s" % (
            return_code, tail(err_buf.text().strip() or out_buf.text().strip()))
    try:
        return json.loads(out_buf.text()), None
    except json.JSONDecodeError as exc:
        return None, "could not parse --capabilities output: %s" % exc


def run_inference(validated, infer_bin):
    try:
        rundir = tempfile.mkdtemp(prefix=RUN_DIR_PREFIX, dir=validated["workdir"])
    except OSError as exc:
        return {"error": "could not create run directory: %s" % exc, "flag": "--workdir"}
    argv = build_argv(validated, infer_bin)
    try:
        return_code, timed_out, stdout, stderr = spawn_bounded(
            argv, rundir, validated["timeout_s"])
    except FileNotFoundError:
        return {"error": "neuriplo-infer binary not found: %s" % infer_bin,
                "flag": "--infer-bin", "run_dir": rundir}
    except OSError as exc:
        return {"error": "could not launch neuriplo-infer: %s" % exc,
                "flag": "--infer-bin", "run_dir": rundir}
    report, report_error = read_report(rundir)
    outcome = {
        "exit_code": return_code,
        "run_dir": rundir,
        "run_report": report,
        "report_error": report_error,
        "artifact": find_artifact(rundir),
        "stdout_tail": tail(stdout),
        "stderr_tail": tail(stderr),
    }
    if timed_out:
        outcome["error"] = "run timed out after %d s" % validated["timeout_s"]
        outcome["flag"] = "--timeout_s"
    return outcome


# Request-id policy (R-H10). Established by inspecting the authoritative
# specs: JSON-RPC 2.0 (https://www.jsonrpc.org/specification, section 4)
# and MCP (lifecycle 2025-11-25 for the served legacy initialize
# handshake; the 2025-11-25 revision explicitly forbids null request ids,
# as does base protocol 2026-07-28: requests MUST carry a string or integer
# id and MUST NOT be null, and error replies echo the request id except
# when it could not be read). A present null id is therefore a structurally
# invalid request (-32600), never a dispatch; only an absent id marks a
# notification (stay silent). Accepted: str, int. Rejected: None, bool
# (in Python bool subclasses int, but JSON true/false is Boolean, not
# Number), float (all floats, even integral ones, since fractional Numbers
# are discouraged and MCP wants integers), and every other type (lists,
# dicts, ...).
def is_valid_request_id(value):
    if isinstance(value, str):
        return True
    return type(value) is int


def handle_request(message, infer_bin):
    if not isinstance(message, dict):
        return error_response(None, -32600, "Invalid Request: message must be an object")
    if message.get("jsonrpc") != "2.0":
        # Wrong, missing, or non-string jsonrpc: never dispatch. Echo the
        # id when it is itself usable; otherwise report a null id.
        request_id = message.get("id")
        if "id" not in message or not is_valid_request_id(request_id):
            request_id = None
        return error_response(request_id, -32600,
                              "Invalid Request: jsonrpc must be exactly \"2.0\"")
    if "id" in message and not is_valid_request_id(message["id"]):
        return error_response(None, -32600,
                              "Invalid Request: id must be a string or integer")
    method = message.get("method")
    if not isinstance(method, str):
        if "id" in message:
            return error_response(message["id"], -32600, "Invalid Request: missing method")
        # Id-less and method-less: not a valid Request object and not a
        # valid notification either, so answer instead of staying silent.
        return error_response(None, -32600, "Invalid Request: missing method")
    if "id" not in message:
        return None  # valid notification (e.g. notifications/initialized): stay silent
    request_id = message["id"]
    params = message.get("params", {})

    if method == "initialize":
        if not isinstance(params, dict):
            return error_response(request_id, -32602, "Invalid params: params must be an object")
        version = negotiate_protocol(params.get("protocolVersion"))
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": ADAPTER_VERSION},
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "tools": [capabilities_tool_schema(), run_tool_schema()]}}
    if method == "server/discover":
        return error_response(
            request_id, -32601,
            "Method not found: server/discover is part of the 2026-07-28 "
            "revision, which this server does not implement; use the legacy "
            "initialize handshake")
    if method == "tools/call":
        if not isinstance(params, dict):
            return error_response(request_id, -32602, "Invalid params: params must be an object")
        name = params.get("name")
        if not isinstance(name, str) or not name:
            return error_response(request_id, -32602, "Invalid params: missing tool name")
        args = params.get("arguments", {})
        if name == "neuriplo_capabilities":
            if args != {}:
                return error_response(request_id, -32602,
                                      "Invalid params: neuriplo_capabilities takes no arguments")
            document, err = run_capabilities(infer_bin)
            if err is not None:
                return {"jsonrpc": "2.0", "id": request_id,
                        "result": tool_text_result({"error": err}, is_error=True)}
            return {"jsonrpc": "2.0", "id": request_id,
                    "result": tool_text_result({"capabilities": document})}
        if name == "neuriplo_run":
            validated, err = validate_run_args(args)
            if err is not None:
                return error_response(
                    request_id, -32602,
                    "Invalid params: %s (%s)" % (err["message"], err["flag"]))
            outcome = run_inference(validated, infer_bin)
            if "error" in outcome and "exit_code" not in outcome:
                return {"jsonrpc": "2.0", "id": request_id,
                        "result": tool_text_result(outcome, is_error=True)}
            failed = outcome.get("exit_code", 1) != 0 or "error" in outcome
            return {"jsonrpc": "2.0", "id": request_id,
                    "result": tool_text_result(outcome, is_error=failed)}
        return error_response(request_id, -32602, "Invalid params: unknown tool '%s'" % name)
    return error_response(request_id, -32601, "Method not found: %s" % method)


def serve(infer_bin):
    stdin = sys.stdin
    stdout = sys.stdout
    while True:
        line = stdin.readline()
        if line == "":
            break  # EOF
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            stdout.write(json.dumps(error_response(None, -32700,
                             "Parse error: %s" % exc)) + "\n")
            stdout.flush()
            continue
        try:
            response = handle_request(message, infer_bin)
        except Exception as exc:  # never let one request kill the server
            log("internal error: %s" % exc)
            if isinstance(message, dict) and "id" not in message:
                response = None  # notification or unparsable: stay silent
            elif (isinstance(message, dict)
                    and is_valid_request_id(message.get("id"))):
                response = error_response(message["id"], -32603,
                                            "Internal error: %s" % exc)
            else:
                response = error_response(None, -32603,
                                            "Internal error: %s" % exc)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main(argv):
    infer_bin = None
    for index, token in enumerate(argv[1:]):
        if token == "--infer-bin" and index + 1 < len(argv[1:]):
            infer_bin = argv[1:][index + 1]
        elif token in ("-h", "--help"):
            sys.stdout.write(__doc__ + "\nUsage: mcp_neuriplo_adapter.py [--infer-bin PATH]\n")
            return 0
    serve(resolve_infer_bin(infer_bin))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
