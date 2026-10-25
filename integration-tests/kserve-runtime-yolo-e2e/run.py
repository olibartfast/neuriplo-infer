#!/usr/bin/env python3
"""Real local KServe runtime E2E validation for the Neuriplo repo cluster."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPOS_ROOT = ROOT.parent


def default_model(neuriplo_infer: Path) -> Path:
    candidates = [
        neuriplo_infer / "models" / "e2e" / "yolo26s.onnx",
        neuriplo_infer / "yolo26s.onnx",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def processed_image_name(model: str, backend: str) -> str:
    # Mirror neuriplo-infer CLICommands.cpp: non-alphanumeric characters in the
    # model/backend tags are replaced with '-'.
    def sanitize(value: str) -> str:
        return "".join(c if c.isalnum() else "-" for c in value)

    return f"processed_{sanitize(model)}_{sanitize(backend)}.png"


def http_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def wait_until_ready(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "runtime did not answer"
    while time.monotonic() < deadline:
        try:
            live = http_json(f"{base_url}/v2/health/live", timeout=2.0)
            ready = http_json(f"{base_url}/v2/health/ready", timeout=2.0)
            if live.get("live") is True and ready.get("ready") is True:
                return
            last_error = f"live={live!r} ready={ready!r}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"runtime did not become ready before timeout: {last_error}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")


def find_infer_binary(neuriplo_infer: Path, build_dir: Path | None, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates: list[Path] = []
    if build_dir is not None:
        candidates.append(build_dir / "app" / "neuriplo-infer")
    candidates.extend([
        neuriplo_infer / "build-kserve-codex" / "app" / "neuriplo-infer",
        neuriplo_infer / "build-kserve" / "app" / "neuriplo-infer",
        neuriplo_infer / "build" / "app" / "neuriplo-infer",
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("missing neuriplo-infer app binary; checked: " + ", ".join(str(p) for p in candidates))


def run_checked(cmd: list[str], cwd: Path, timeout: float) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError("command failed (exit %d): %s\n%s" % (result.returncode, " ".join(cmd), result.stdout))
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument("--model-name", default="yolo")
    parser.add_argument("--task-type", default="yolo26")
    parser.add_argument("--runtime-bin", type=Path, default=REPOS_ROOT / "neuriplo-kserve-runtime" / "build" / "real-onnx" / "neuriplo-kserve-runtime")
    parser.add_argument("--infer-build-dir", type=Path, default=None)
    parser.add_argument("--infer-bin", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=REPOS_ROOT / "neuriplo-infer" / "data" / "dog.jpg")
    parser.add_argument("--labels", type=Path, default=REPOS_ROOT / "neuriplo-infer" / "labels" / "coco.names")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--infer-timeout", type=float, default=120.0)
    args = parser.parse_args()

    neuriplo_infer = REPOS_ROOT / "neuriplo-infer"
    runtime_repo = REPOS_ROOT / "neuriplo-kserve-runtime"
    model = args.model if args.model is not None else default_model(neuriplo_infer)
    infer_bin = find_infer_binary(neuriplo_infer, args.infer_build_dir, args.infer_bin)
    # neuriplo-infer names the rendered image processed_<model>_<backend>.png; the
    # remote path always reports the "kserve" backend and uses --type as the model.
    output_image = neuriplo_infer / "data" / "output" / processed_image_name(args.task_type, "kserve")
    base_url = f"http://127.0.0.1:{args.port}"

    require_file(args.runtime_bin, "neuriplo-kserve-runtime binary")
    require_file(infer_bin, "neuriplo-infer app binary")
    require_file(model, "YOLO ONNX model")
    require_file(args.source, "input image")
    require_file(args.labels, "labels file")

    log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"neuriplo-kserve-e2e-{args.port}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process: subprocess.Popen[str] | None = None
    start_ns = time.time_ns()

    try:
        runtime_cmd = [
            str(args.runtime_bin),
            "--model-name", args.model_name,
            "--model-path", str(model),
            "--backend", "onnx_runtime",
            "--port", str(args.port),
            "--instances", "1",
        ]
        process = subprocess.Popen(
            runtime_cmd,
            cwd=runtime_repo,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        wait_until_ready(base_url, args.startup_timeout)

        metadata = http_json(f"{base_url}/v2/models/{args.model_name}", timeout=5.0)
        if metadata.get("platform") != "neuriplo_onnx_runtime":
            raise RuntimeError(f"unexpected runtime platform metadata: {metadata!r}")
        input_names = {item.get("name") for item in metadata.get("inputs", [])}
        output_names = {item.get("name") for item in metadata.get("outputs", [])}
        if "images" not in input_names or "output0" not in output_names:
            raise RuntimeError(f"metadata missing expected YOLO tensors: {metadata!r}")

        infer_cmd = [
            str(infer_bin),
            f"--type={args.task_type}",
            f"--source={args.source}",
            f"--labels={args.labels}",
            f"--kserve_endpoint={base_url}",
            f"--kserve_model_name={args.model_name}",
            "--kserve_transport=http",
            "--min_confidence=0.25",
        ]
        app_output = run_checked(infer_cmd, cwd=neuriplo_infer, timeout=args.infer_timeout)

        require_file(output_image, "processed output image")
        stat = output_image.stat()
        if stat.st_size <= 0 or stat.st_mtime_ns < start_ns:
            raise RuntimeError(f"output image was not refreshed: {output_image} size={stat.st_size}")

        metrics = urllib.request.urlopen(f"{base_url}/metrics", timeout=5.0).read().decode("utf-8")
        expected_metric = f'neuriplo_http_infer_requests_success_total{{model="{args.model_name}",version="1"}} 1'
        if expected_metric not in metrics:
            raise RuntimeError(f"missing success metric: {expected_metric}")
        failure_metric = f'neuriplo_http_infer_requests_failure_total{{model="{args.model_name}",version="1"}} 0'
        if failure_metric not in metrics:
            raise RuntimeError(f"failure metric is not zero: {failure_metric}")

        print("kserve runtime e2e ok")
        print(f"runtime log: {log_path}")
        print(f"app binary: {infer_bin}")
        print(f"output image: {output_image} ({stat.st_size} bytes)")
        if app_output.strip():
            print("app output tail:")
            print("\n".join(app_output.strip().splitlines()[-12:]))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"runtime log: {log_path}", file=sys.stderr)
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
