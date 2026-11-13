#!/usr/bin/env python3
"""Unit tests for scripts/mcp_neuriplo_adapter.py.

All tests use a fake neuriplo-infer executable: no GPU, no model download,
no sibling checkout. A subset drives the adapter over stdio exactly as an MCP
host would (initialize -> tools/list -> tools/call).
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


ADAPTER = Path(__file__).with_name("mcp_neuriplo_adapter.py")


def load_module():
    spec = importlib.util.spec_from_file_location("mcp_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module()

FAKE_BIN = """#!/usr/bin/env python3
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
if "--capabilities" in args:
    cap_marker = os.environ.get("NEURIPLO_MCP_TEST_CAP_MARKER")
    if cap_marker:
        with open(cap_marker, "a") as fh:
            fh.write("spawn\\n")
    cap_mode = os.environ.get("NEURIPLO_MCP_TEST_CAP_MODE", "")
    if cap_mode == "noisy":
        sys.stdout.write('{"schema_version": 2, "data": "')
        sys.stdout.write("x" * (2 * 1024 * 1024))
        sys.stdout.write('"}')
        sys.stdout.flush()
        sys.exit(0)
    if cap_mode == "truncated":
        sys.stdout.write('{"schema_version": 2, "producer": ')
        sys.stdout.flush()
        sys.exit(0)
    if cap_mode == "slow":
        time.sleep(30)
        sys.exit(0)
    print(json.dumps({"schema_version": 2,
                      "producer": {"name": "neuriplo-infer", "version": "test"},
                      "execution": {"workflows": [{"id": "local"}]}}))
    sys.exit(0)
marker = os.environ.get("NEURIPLO_MCP_TEST_MARKER")
if marker:
    with open(marker, "a") as fh:
        fh.write("spawn\\n")
atype = ""
for index, token in enumerate(args):
    if token == "--type" and index + 1 < len(args):
        atype = args[index + 1]
    elif token.startswith("--type="):
        atype = token[len("--type="):]
if atype == "failme":
    sys.stderr.write("boom: simulated backend failure\\n")
    sys.exit(1)
if atype == "sleepy":
    time.sleep(30)
    sys.exit(0)
if atype == "orphan":
    # Grandchild inherits stdout/stderr (fds 1,2) and outlives the child,
    # holding the adapter's pipes open past child exit.
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])
    pidfile = os.environ.get("NEURIPLO_MCP_TEST_PIDFILE")
    if pidfile:
        with open(pidfile, "w") as fh:
            fh.write(str(grandchild.pid))
    print("ARGV:" + json.dumps(args))
    sys.exit(0)
print("ARGV:" + json.dumps(args))
if atype == "noisy":
    sys.stdout.write("x" * (5 * 1024 * 1024))
    print("NOISY-DONE")
if atype != "noreport":
    out = Path(os.getcwd()) / "data" / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_report.json").write_text(json.dumps(
        {"schema_version": 1, "status": "success", "stage": "render"}))
    (out / "processed_test_local.png").write_text("fake-png")
print("Inference time: 1 ms")
sys.exit(0)
"""


class AdapterProcess:
    def __init__(self, env_extra=None):
        env = dict(os.environ)
        env.update(env_extra or {})
        self.proc = subprocess.Popen(
            [sys.executable, str(ADAPTER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, env=env, bufsize=1,
        )

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def send_raw(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def recv(self):
        line = self.proc.stdout.readline()
        if not line:
            raise AssertionError("adapter closed stdout unexpectedly")
        return json.loads(line)

    def call(self, tool, args, request_id="1"):
        self.send({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}})
        return self.recv()

    def close(self):
        try:
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        finally:
            try:
                self.proc.terminate()
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10)
            finally:
                try:
                    self.proc.stdout.close()
                except Exception:
                    pass


def stdio_call(server, method, params=None, request_id="1"):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    server.send(message)
    return server.recv()


class StdioFramingTest(unittest.TestCase):
    def test_initialize_and_list(self):
        server = AdapterProcess()
        try:
            response = stdio_call(server, "initialize",
                                  {"protocolVersion": "2025-11-25",
                                   "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0"}})
            self.assertEqual(response["result"]["protocolVersion"], "2025-11-25")
            self.assertEqual(response["result"]["serverInfo"]["name"], "neuriplo-mcp")
            self.assertEqual(response["result"]["serverInfo"]["version"], "0.2.1")
            server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            response = stdio_call(server, "tools/list", {}, request_id="2")
            names = [tool["name"] for tool in response["result"]["tools"]]
            self.assertEqual(names, ["neuriplo_capabilities", "neuriplo_run"])
        finally:
            server.close()

    def test_initialize_without_protocol_version_gets_default(self):
        server = AdapterProcess()
        try:
            response = stdio_call(server, "initialize", {"capabilities": {}})
            self.assertEqual(response["result"]["protocolVersion"], "2025-11-25")
        finally:
            server.close()

    def test_malformed_json_does_not_kill_server(self):
        server = AdapterProcess()
        try:
            server.send_raw("{bad json")
            response = server.recv()
            self.assertEqual(response["error"]["code"], -32700)
            response = stdio_call(server, "ping", {}, request_id="9")
            self.assertEqual(response["result"], {})
        finally:
            server.close()

    def test_oversized_integer_id_is_parse_error_and_server_survives(self):
        server = AdapterProcess()
        try:
            server.send_raw('{"jsonrpc":"2.0","id":' + "1" * 5000 +
                            ',"method":"ping"}\n')
            response = server.recv()
            self.assertEqual(response["error"]["code"], -32700)
            followup = stdio_call(server, "ping", {}, request_id="after-large-id")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_missing_method_is_invalid_request(self):
        server = AdapterProcess()
        try:
            server.send({"jsonrpc": "2.0", "id": "7"})
            response = server.recv()
            self.assertEqual(response["error"]["code"], -32600)
        finally:
            server.close()

    def test_null_id_rejected(self):
        # The served 2025-11-25 revision forbids null request ids: a present
        # null id is an invalid request (-32600), never a dispatch.
        server = AdapterProcess()
        try:
            server.send({"jsonrpc": "2.0", "id": None, "method": "ping", "params": {}})
            response = server.recv()
            self.assertIsNone(response["id"])
            self.assertEqual(response["error"]["code"], -32600)
            followup = stdio_call(server, "ping", {}, request_id="1")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_absent_id_is_notification_silence(self):
        server = AdapterProcess()
        try:
            server.send({"jsonrpc": "2.0", "method": "ping", "params": {}})
            response = stdio_call(server, "ping", {}, request_id="8")
            self.assertEqual(response["id"], "8")
        finally:
            server.close()

    def test_server_discover_answered_as_legacy_only(self):
        server = AdapterProcess()
        try:
            response = stdio_call(server, "server/discover", {})
            self.assertEqual(response["error"]["code"], -32601)
            self.assertIn("legacy", response["error"]["message"])
        finally:
            server.close()

    def test_unknown_method(self):
        server = AdapterProcess()
        try:
            response = stdio_call(server, "resources/list", {})
            self.assertEqual(response["error"]["code"], -32601)
        finally:
            server.close()


class EnvelopeValidationTest(unittest.TestCase):
    def test_jsonrpc_missing_is_invalid_request(self):
        server = AdapterProcess()
        try:
            server.send({"id": "1", "method": "ping", "params": {}})
            response = server.recv()
            self.assertEqual(response["error"]["code"], -32600)
            self.assertEqual(response["id"], "1")
            followup = stdio_call(server, "ping", {}, request_id="2")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_jsonrpc_wrong_value_is_invalid_request(self):
        server = AdapterProcess()
        try:
            server.send({"jsonrpc": "1.0", "id": "1",
                         "method": "ping", "params": {}})
            response = server.recv()
            self.assertEqual(response["error"]["code"], -32600)
            self.assertEqual(response["id"], "1")
            followup = stdio_call(server, "ping", {}, request_id="2")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_jsonrpc_wrong_type_is_invalid_request(self):
        server = AdapterProcess()
        try:
            server.send({"jsonrpc": 2.0, "id": "1",
                         "method": "ping", "params": {}})
            response = server.recv()
            self.assertEqual(response["error"]["code"], -32600)
            followup = stdio_call(server, "ping", {}, request_id="2")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_invalid_message_without_id_still_gets_response(self):
        server = AdapterProcess()
        try:
            # No jsonrpc and no id: not a valid notification, so the
            # server must answer -32600 with a null id, not stay silent.
            server.send({"method": "ping"})
            response = server.recv()
            self.assertEqual(response["error"]["code"], -32600)
            self.assertIsNone(response["id"])
            followup = stdio_call(server, "ping", {}, request_id="2")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_invalid_id_types_are_invalid_request(self):
        server = AdapterProcess()
        try:
            for bad_id in (True, False, 1.5, 1.0, ["1"], {"id": 1}):
                with self.subTest(bad_id=bad_id):
                    server.send({"jsonrpc": "2.0", "id": bad_id,
                                 "method": "ping", "params": {}})
                    response = server.recv()
                    self.assertEqual(response["error"]["code"], -32600)
                    self.assertIsNone(response["id"])
            followup = stdio_call(server, "ping", {}, request_id="2")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_integer_id_is_accepted(self):
        server = AdapterProcess()
        try:
            response = stdio_call(server, "ping", {}, request_id=42)
            self.assertEqual(response["id"], 42)
            self.assertEqual(response["result"], {})
        finally:
            server.close()

    def test_older_protocol_version_still_negotiates(self):
        server = AdapterProcess()
        try:
            for version in ("2024-11-05", "2025-03-26", "2025-06-18"):
                with self.subTest(version=version):
                    response = stdio_call(server, "initialize",
                                          {"protocolVersion": version},
                                          request_id=version)
                    self.assertEqual(
                        response["result"]["protocolVersion"], version)
        finally:
            server.close()


class CapabilitiesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fake = Path(self.temporary.name) / "fake-infer"
        self.fake.write_text(FAKE_BIN)
        self.fake.chmod(0o755)

    def test_capabilities_passthrough(self):
        expected_raw = subprocess.run(
            [str(self.fake), "--capabilities"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=10).stdout
        expected_doc = json.loads(expected_raw)
        server = AdapterProcess(env_extra={"NEURIPLO_INFER_BIN": str(self.fake)})
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_capabilities", {}, request_id="3")
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertEqual(payload["capabilities"]["schema_version"], 2)
            self.assertEqual(
                payload["capabilities"]["producer"]["name"], "neuriplo-infer")
            self.assertEqual(payload, {"capabilities": expected_doc})
            self.assertEqual(
                response["result"]["content"][0]["text"],
                json.dumps({"capabilities": expected_doc}, indent=2))
        finally:
            server.close()

    def test_capabilities_missing_binary_is_tool_error(self):
        server = AdapterProcess(env_extra={"NEURIPLO_INFER_BIN": "/does-not-exist"})
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_capabilities", {}, request_id="3")
            self.assertTrue(response["result"].get("isError"))
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertIn("not found", payload["error"])
        finally:
            server.close()

    def test_capabilities_rejects_arguments_without_spawning(self):
        marker = Path(self.temporary.name) / "marker"
        cap_marker = Path(self.temporary.name) / "cap-marker"
        server = AdapterProcess(env_extra={"NEURIPLO_INFER_BIN": str(self.fake),
                                           "NEURIPLO_MCP_TEST_MARKER": str(marker),
                                           "NEURIPLO_MCP_TEST_CAP_MARKER": str(cap_marker)})
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_capabilities",
                                   {"infer_bin": "/tmp/x"}, request_id="3")
            self.assertEqual(response["error"]["code"], -32602)
            self.assertFalse(marker.exists())
            self.assertFalse(cap_marker.exists())
        finally:
            server.close()

    def test_capabilities_rejects_null_arguments_without_spawning(self):
        marker = Path(self.temporary.name) / "cap-marker"
        server = AdapterProcess(env_extra={
            "NEURIPLO_INFER_BIN": str(self.fake),
            "NEURIPLO_MCP_TEST_CAP_MARKER": str(marker)})
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_capabilities", None, request_id="3")
            self.assertEqual(response["error"]["code"], -32602)
            self.assertIn("takes no arguments", response["error"]["message"])
            self.assertFalse(marker.exists())
        finally:
            server.close()

    def test_capabilities_noisy_is_oversized_error(self):
        server = AdapterProcess(env_extra={"NEURIPLO_INFER_BIN": str(self.fake),
                                           "NEURIPLO_MCP_TEST_CAP_MODE": "noisy"})
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_capabilities", {}, request_id="3")
            self.assertTrue(response["result"].get("isError"))
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertIn("error", payload)
            self.assertNotIn("capabilities", payload)
            self.assertIn("1048576", payload["error"])
            self.assertIn("limit", payload["error"])
            self.assertNotIn("could not parse", payload["error"])
            followup = stdio_call(server, "ping", {}, request_id="4")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_capabilities_truncated_is_parse_error(self):
        server = AdapterProcess(env_extra={"NEURIPLO_INFER_BIN": str(self.fake),
                                           "NEURIPLO_MCP_TEST_CAP_MODE": "truncated"})
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_capabilities", {}, request_id="3")
            self.assertTrue(response["result"].get("isError"))
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertIn("could not parse", payload["error"])
            self.assertNotIn("1048576", payload["error"])
            self.assertNotIn("exceeded", payload["error"])
            followup = stdio_call(server, "ping", {}, request_id="4")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_capabilities_slow_times_out(self):
        server = AdapterProcess(env_extra={"NEURIPLO_INFER_BIN": str(self.fake),
                                           "NEURIPLO_MCP_TEST_CAP_MODE": "slow",
                                           "NEURIPLO_MCP_CAPABILITIES_TIMEOUT_S": "1"})
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_capabilities", {}, request_id="3")
            self.assertTrue(response["result"].get("isError"))
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertIn("timed out", payload["error"])
            followup = stdio_call(server, "ping", {}, request_id="4")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()


class RunTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.fake = root / "fake-infer"
        self.fake.write_text(FAKE_BIN)
        self.fake.chmod(0o755)
        self.marker = root / "marker"
        self.env = {"NEURIPLO_INFER_BIN": str(self.fake),
                    "NEURIPLO_MCP_TEST_MARKER": str(self.marker)}
        self.dir_a = root / "assets"
        self.dir_a.mkdir()
        self.image = self.dir_a / "im age;$(touch).jpg"
        self.image.write_bytes(b"\xff\xd8fake")
        self.weights = self.dir_a / "model.onnx"
        self.weights.write_bytes(b"fake-weights")
        self.dir_b = root / "workroot"
        self.dir_b.mkdir()

    def spawns(self):
        if not self.marker.exists():
            return 0
        return len(self.marker.read_text().strip().splitlines())

    def test_run_success_returns_report_and_artifact(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "yolo", "source": str(self.image),
                "weights": str(self.weights), "workdir": str(self.dir_b)}, request_id="4")
            self.assertNotIn("isError", response["result"])
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertEqual(payload["exit_code"], 0)
            self.assertIsNone(payload["report_error"])
            self.assertEqual(payload["run_report"]["status"], "success")
            self.assertTrue(payload["run_dir"].startswith(str(self.dir_b)))
            self.assertTrue(payload["artifact"].startswith(payload["run_dir"]))
            self.assertTrue(payload["artifact"].endswith("processed_test_local.png"))
            self.assertIn("Inference time", payload["stdout_tail"])
        finally:
            server.close()

    def test_default_run_dir_is_system_temp(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "yolo", "source": str(self.image),
                "weights": str(self.weights)}, request_id="4")
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertTrue(payload["run_dir"].startswith(tempfile.gettempdir()))
        finally:
            server.close()

    def test_prior_success_not_attributed_to_later_failure(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            good = server.call("neuriplo_run", {
                "type": "yolo", "source": str(self.image),
                "weights": str(self.weights)}, request_id="4")
            run_a = json.loads(good["result"]["content"][0]["text"])["run_dir"]
            bad = server.call("neuriplo_run", {
                "type": "failme", "source": str(self.image),
                "weights": str(self.weights)}, request_id="5")
            self.assertTrue(bad["result"].get("isError"))
            payload = json.loads(bad["result"]["content"][0]["text"])
            self.assertEqual(payload["exit_code"], 1)
            self.assertNotEqual(payload["run_dir"], run_a)
            self.assertIsNone(payload["run_report"])
            self.assertIsNotNone(payload["report_error"])
            self.assertIn("no run report", payload["report_error"])
            self.assertIsNone(payload["artifact"])
            self.assertNotIn(run_a, json.dumps(payload))
            self.assertIn("boom", payload["stderr_tail"])
        finally:
            server.close()

    def test_success_without_report_is_explicit_not_silent(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "noreport", "source": str(self.image),
                "weights": str(self.weights)}, request_id="4")
            self.assertNotIn("isError", response["result"])
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertEqual(payload["exit_code"], 0)
            self.assertIsNone(payload["run_report"])
            self.assertIsNotNone(payload["report_error"])
            self.assertIsNone(payload["artifact"])
        finally:
            server.close()

    def test_argv_uses_equals_form(self):
        # Load-bearing: this binary's OpenCV parser misparses space-separated
        # valued flags (source reads as "true"); the repo's own scripts and
        # app tests use --key=value, so the adapter must too.
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "yolo", "source": str(self.image),
                "weights": str(self.weights), "use_gpu": True}, request_id="4")
            payload = json.loads(response["result"]["content"][0]["text"])
            argv_line = next(line for line in payload["stdout_tail"].splitlines()
                             if line.startswith("ARGV:"))
            argv = json.loads(argv_line[len("ARGV:"):])
            self.assertIn("--type=yolo", argv)
            self.assertIn("--source=%s" % self.image, argv)
            self.assertIn("--weights=%s" % self.weights, argv)
            self.assertIn("--no_display", argv)
            self.assertIn("--use-gpu", argv)
            self.assertNotIn("--source", argv)
        finally:
            server.close()

    def test_paths_resolved_absolute_across_directories(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "yolo", "source": str(self.image),
                "weights": str(self.weights), "workdir": str(self.dir_b)}, request_id="4")
            payload = json.loads(response["result"]["content"][0]["text"])
            argv_line = next(line for line in payload["stdout_tail"].splitlines()
                             if line.startswith("ARGV:"))
            argv = json.loads(argv_line[len("ARGV:"):])
            self.assertIn("--source=%s" % self.image, argv)
            source_value = next(t[len("--source="):] for t in argv
                                if t.startswith("--source="))
            self.assertTrue(os.path.isabs(source_value))
        finally:
            server.close()

    def test_missing_source_names_the_flag(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "yolo", "source": "/does-not-exist.jpg",
                "weights": str(self.weights)}, request_id="4")
            self.assertEqual(response["error"]["code"], -32602)
            self.assertIn("--source", response["error"]["message"])
            self.assertEqual(self.spawns(), 0)
        finally:
            server.close()

    def test_kserve_params_rejected_as_out_of_scope(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "yolo", "source": str(self.image),
                "weights": str(self.weights),
                "kserve_endpoint": "http://127.0.0.1:8080"}, request_id="4")
            self.assertEqual(response["error"]["code"], -32602)
            self.assertIn("MVP scope", response["error"]["message"])
            self.assertEqual(self.spawns(), 0)
        finally:
            server.close()

    def test_unknown_key_rejected(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "yolo", "source": str(self.image),
                "weights": str(self.weights), "bogus_flag": 1}, request_id="4")
            self.assertEqual(response["error"]["code"], -32602)
            self.assertIn("bogus_flag", response["error"]["message"])
            self.assertEqual(self.spawns(), 0)
        finally:
            server.close()

    def test_non_image_source_rejected_without_spawning(self):
        video = self.dir_a / "clip.mpeg"
        video.write_bytes(b"video")
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            for suffix in (".mpeg", ".mpg", ".ts", ".unknown"):
                source = self.dir_a / ("clip" + suffix)
                source.touch()
                response = server.call("neuriplo_run", {
                    "type": "yolo", "source": str(source),
                    "weights": str(self.weights)}, request_id=suffix)
                self.assertEqual(response["error"]["code"], -32602)
                self.assertIn("still image", response["error"]["message"])
            self.assertEqual(self.spawns(), 0)
        finally:
            server.close()

    def test_strict_types_rejected_without_spawning(self):
        cases = [
            ({"use_gpu": "false"}, "--use_gpu"),
            ({"use_gpu": 1}, "--use_gpu"),
            ({"min_confidence": True}, "--min_confidence"),
            ({"nms_threshold": "0.5"}, "--nms_threshold"),
            ({"min_confidence": 2}, "--min_confidence"),
            ({"labels": 123}, "--labels"),
            ({"labels": None}, "--labels"),
            ({"workdir": 123}, "--workdir"),
            ({"workdir": None}, "--workdir"),
            ({"min_confidence": None}, "--min_confidence"),
            ({"nms_threshold": None}, "--nms_threshold"),
            ({"timeout_s": "300"}, "--timeout_s"),
            ({"timeout_s": True}, "--timeout_s"),
            ({"timeout_s": 0}, "--timeout_s"),
        ]
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            for index, (extra, flag) in enumerate(cases):
                with self.subTest(extra=extra):
                    args = {"type": "yolo", "source": str(self.image),
                            "weights": str(self.weights)}
                    args.update(extra)
                    response = server.call("neuriplo_run", args,
                                           request_id="case-%d" % index)
                    self.assertEqual(response["error"]["code"], -32602)
                    self.assertIn(flag, response["error"]["message"])
            self.assertEqual(self.spawns(), 0)
        finally:
            server.close()

    def test_binary_failure_is_tool_error_with_tails(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "failme", "source": str(self.image),
                "weights": str(self.weights), "workdir": str(self.dir_b)}, request_id="4")
            self.assertTrue(response["result"].get("isError"))
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertEqual(payload["exit_code"], 1)
            self.assertIn("boom", payload["stderr_tail"])
        finally:
            server.close()

    def test_timeout_kills_child_and_stays_alive(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "sleepy", "source": str(self.image),
                "weights": str(self.weights), "timeout_s": 1}, request_id="4")
            self.assertTrue(response["result"].get("isError"))
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertIn("timed out", payload["error"])
            self.assertIn("--timeout_s", payload.get("flag", ""))
            followup = stdio_call(server, "ping", {}, request_id="5")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_orphan_grandchild_reaped_no_lingering_readers(self):
        pidfile = Path(self.temporary.name) / "grandchild.pid"
        env = dict(self.env)
        env["NEURIPLO_MCP_TEST_PIDFILE"] = str(pidfile)
        server = AdapterProcess(env_extra=env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            start = time.monotonic()
            response = server.call("neuriplo_run", {
                "type": "orphan", "source": str(self.image),
                "weights": str(self.weights)}, request_id="4")
            elapsed = time.monotonic() - start
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertEqual(payload["exit_code"], 0)
            self.assertIsNotNone(payload["report_error"])
            self.assertIsNone(payload["artifact"])
            # Prompt response: without group cleanup the lingering readers
            # would stall on the grandchild-held pipes.
            self.assertLess(elapsed, 15)
            grandchild = int(pidfile.read_text().strip())
            for _ in range(50):
                try:
                    os.kill(grandchild, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                self.fail("orphan grandchild survived cleanup")
            lingering = [thread for thread in threading.enumerate()
                        if thread.name.startswith("mcp-drain-")
                        and thread.is_alive()]
            self.assertEqual(lingering, [])
        finally:
            server.close()

    def test_noisy_child_output_stays_bounded(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = server.call("neuriplo_run", {
                "type": "noisy", "source": str(self.image),
                "weights": str(self.weights)}, request_id="4")
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertEqual(payload["exit_code"], 0)
            self.assertLessEqual(len(payload["stdout_tail"]), 4000)
            self.assertIn("NOISY-DONE", payload["stdout_tail"])
            followup = stdio_call(server, "ping", {}, request_id="5")
            self.assertEqual(followup["result"], {})
        finally:
            server.close()

    def test_unknown_tool(self):
        server = AdapterProcess(env_extra=self.env)
        try:
            stdio_call(server, "initialize", {"protocolVersion": "2025-11-25"})
            response = stdio_call(server, "tools/call",
                                  {"name": "nope", "arguments": {}}, request_id="8")
            self.assertEqual(response["error"]["code"], -32602)
        finally:
            server.close()


class PureValidationTest(unittest.TestCase):
    def test_no_new_dependency_imports(self):
        stdlib = {"json", "os", "shutil", "signal", "subprocess", "sys", "tempfile",
                  "threading", "pathlib", "__future__"}
        source = ADAPTER.read_text(encoding="utf-8")
        imported = set()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                for token in stripped[len("import "):].split(","):
                    imported.add(token.strip().split(".")[0])
            elif stripped.startswith("from "):
                imported.add(stripped.split()[1].split(".")[0])
        self.assertLessEqual(imported, stdlib | {"annotations"})


if __name__ == "__main__":
    unittest.main()
