#!/usr/bin/env python3
"""Real local KServe runtime E2E for EdgeCrafter detection across backends.

Mirrors the YOLO ``kserve-runtime-yolo-e2e`` runner but exercises the EdgeCrafter
``ecdet`` dual-input contract (``images`` FP32 + ``orig_target_sizes`` INT64 ->
``labels`` INT64, ``boxes``/``scores`` FP32). It also asserts the advertised
datatypes, which is the regression guard for the metadata dtype-propagation fix:
without it the INT64 tensors were reported (and served) as FP32.

All model artifacts are read from per-backend Triton-style model folders named
``<model>_<size>_<backend>``:

  ${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_onnx/1/model.onnx
  ${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_tensorrt/1/model.engine
  ${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_openvino/1/model.{xml,bin}
  ${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_executorch/1/model.pte

Prepare the repository with ``prepare_model_repository.py`` when files are
missing.

Select the serving backend with ``--backend`` (``onnx_runtime``, ``tensorrt``,
``openvino``, or ``executorch``). Use ``--mode local`` for in-process inference
(without a KServe hop) on ``openvino`` or ``executorch``, or ``--mode kserve``
with ``--transport {http,grpc}``. ``run_openvino_matrix.py`` and
``run_executorch_matrix.py`` run local + HTTP + gRPC in one invocation.
"""

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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from repository_layout import backend_model_name, model_version_dir

REPOS_ROOT = ROOT.parent
DEFAULT_MODEL_REPOSITORY = Path(
    os.environ.get("NEURIPLO_MODEL_REPOSITORY", Path.home() / "model_repository")
)
LOCAL_BACKENDS = frozenset({"openvino", "executorch"})
LOCAL_OUTPUT_BACKEND_LABELS = {
    "openvino": ("openvino", "local"),
    "executorch": ("executorch", "local"),
}

# Per-backend defaults: runtime build dir(s) and artifact filename under ecdet_s_<backend>/1/.
BACKEND_DEFAULTS = {
    "onnx_runtime": {
        "build": "real-onnx",
        "build_grpc": "real-onnx-grpc",
        "artifact": "model.onnx",
    },
    "tensorrt": {
        "build": "real-trt",
        "build_grpc": "real-trt-grpc",
        "artifact": "model.engine",
    },
    "openvino": {
        "build": "real-openvino",
        "build_grpc": "real-openvino-grpc",
        "artifact": "model.xml",
        "artifact_fallback": "model.onnx",
    },
    "executorch": {
        "build": "real-executorch",
        "build_grpc": "real-executorch-grpc",
        "artifact": "model.pte",
    },
}


def processed_image_name(model: str, backend: str) -> str:
    def sanitize(value: str) -> str:
        return "".join(c if c.isalnum() else "-" for c in value)

    return f"processed_{sanitize(model)}_{sanitize(backend)}.png"


def resolve_model_artifact(backend: str, model_repository: Path) -> Path:
    cfg = BACKEND_DEFAULTS[backend]
    version_dir = model_version_dir(model_repository, backend)
    primary = version_dir / cfg["artifact"]
    if primary.is_file():
        return primary
    fallback_name = cfg.get("artifact_fallback")
    if fallback_name:
        fallback = version_dir / fallback_name
        if fallback.is_file():
            return fallback
        if backend == "openvino" and fallback_name == "model.onnx":
            shared_onnx = model_version_dir(model_repository, "onnx_runtime") / fallback_name
            if shared_onnx.is_file():
                return shared_onnx
    return primary


def default_runtime_bin(backend: str, transport: str) -> Path:
    cfg = BACKEND_DEFAULTS[backend]
    build = cfg["build_grpc"] if transport == "grpc" else cfg["build"]
    return REPOS_ROOT / "neuriplo-kserve-runtime" / "build" / build / "neuriplo-kserve-runtime"


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


def find_kserve_infer_binary(
    neuriplo_infer: Path,
    build_dir: Path | None,
    explicit: Path | None,
) -> Path:
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


def find_local_infer_binary(
    neuriplo_infer: Path,
    backend: str,
    build_dir: Path | None,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        return explicit
    candidates: list[Path] = []
    if build_dir is not None:
        candidates.append(build_dir / "app" / "neuriplo-infer")
    if backend == "openvino":
        candidates.extend([
            neuriplo_infer / "build-openvino" / "app" / "neuriplo-infer",
            neuriplo_infer / "build" / "app" / "neuriplo-infer",
        ])
    elif backend == "executorch":
        candidates.extend([
            neuriplo_infer / "build-executorch" / "app" / "neuriplo-infer",
            neuriplo_infer / "build" / "app" / "neuriplo-infer",
        ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        f"missing local neuriplo-infer binary for {backend}; checked: "
        + ", ".join(str(p) for p in candidates)
    )


def prepend_ld_library_path(env: dict[str, str], lib_dirs: list[Path]) -> None:
    existing = env.get("LD_LIBRARY_PATH", "")
    extra = os.pathsep.join(str(d) for d in lib_dirs if d.is_dir())
    if extra:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (extra, existing) if p)


def default_openvino_root() -> Path:
    if "OPENVINO_DIR" in os.environ:
        return Path(os.environ["OPENVINO_DIR"])
    return Path.home() / "dependencies" / "openvino_2025.2.0"


def default_executorch_root() -> Path:
    if "EXECUTORCH_DIR" in os.environ:
        return Path(os.environ["EXECUTORCH_DIR"])
    return Path.home() / "dependencies" / "executorch"


def runtime_env(
    backend: str,
    tensorrt_lib_dirs: list[Path],
    openvino_root: Path,
    executorch_root: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    if backend == "tensorrt":
        prepend_ld_library_path(env, tensorrt_lib_dirs)
    elif backend == "openvino":
        prepend_ld_library_path(env, [openvino_root / "runtime" / "lib" / "intel64"])
    elif backend == "executorch":
        prepend_ld_library_path(env, [executorch_root / "lib"])
    return env


def run_checked(cmd: list[str], cwd: Path, timeout: float, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError("command failed (exit %d): %s\n%s" % (result.returncode, " ".join(cmd), result.stdout))
    return result.stdout


def metadata_datatypes(metadata: dict[str, Any]) -> dict[str, str]:
    types: dict[str, str] = {}
    for item in metadata.get("inputs", []) + metadata.get("outputs", []):
        types[item.get("name")] = item.get("datatype")
    return types


def assert_edgecrafter_metadata(metadata: dict[str, Any], backend: str) -> None:
    expected_platform = "neuriplo_" + backend
    if metadata.get("platform") != expected_platform:
        raise RuntimeError(f"unexpected runtime platform metadata: {metadata!r}")

    input_names = {item.get("name") for item in metadata.get("inputs", [])}
    output_names = {item.get("name") for item in metadata.get("outputs", [])}
    if not {"images", "orig_target_sizes"} <= input_names:
        raise RuntimeError(f"metadata missing expected EdgeCrafter inputs: {metadata!r}")
    if not {"labels", "boxes", "scores"} <= output_names:
        raise RuntimeError(f"metadata missing expected EdgeCrafter outputs: {metadata!r}")

    datatypes = metadata_datatypes(metadata)
    expected_datatypes = {
        "images": "FP32",
        "orig_target_sizes": "INT64",
        "labels": "INT64",
        "boxes": "FP32",
        "scores": "FP32",
    }
    for name, expected in expected_datatypes.items():
        if datatypes.get(name) != expected:
            raise RuntimeError(
                f"datatype mismatch for {name!r}: expected {expected}, got {datatypes.get(name)!r} "
                f"(metadata: {metadata!r})"
            )


def prepare_model_repository(
    backend: str,
    model_repository: Path,
    skip_openvino: bool,
    skip_executorch: bool,
) -> None:
    script = Path(__file__).resolve().parent / "prepare_model_repository.py"
    cmd = [
        sys.executable,
        str(script),
        "--model-repository",
        str(model_repository),
        "--backend",
        backend,
    ]
    if skip_openvino:
        cmd.append("--skip-openvino")
    if skip_executorch:
        cmd.append("--skip-executorch")
    run_checked(cmd, cwd=ROOT, timeout=900.0)


def run_e2e(args: argparse.Namespace) -> int:
    neuriplo_infer = REPOS_ROOT / "neuriplo-infer"
    runtime_repo = REPOS_ROOT / "neuriplo-kserve-runtime"
    model_repository = args.model_repository
    model = args.model if args.model is not None else resolve_model_artifact(args.backend, model_repository)
    start_ns = time.time_ns()

    if args.mode == "local":
        if args.backend not in LOCAL_BACKENDS:
            raise RuntimeError(f"--mode local is supported only with --backend {' or '.join(sorted(LOCAL_BACKENDS))}")
        infer_bin = find_local_infer_binary(
            neuriplo_infer, args.backend, args.local_infer_build_dir, args.local_infer_bin
        )
        output_candidates = [
            neuriplo_infer / "data" / "output" / processed_image_name(args.task_type, label)
            for label in LOCAL_OUTPUT_BACKEND_LABELS[args.backend]
        ]
        require_file(infer_bin, "local neuriplo-infer app binary (OpenVINO build)")
        require_file(model, f"EdgeCrafter {args.backend} model artifact")
        require_file(args.source, "input image")
        require_file(args.labels, "labels file")

        infer_cmd = [
            str(infer_bin),
            f"--type={args.task_type}",
            f"--source={args.source}",
            f"--labels={args.labels}",
            f"--weights={model}",
            "--min_confidence=0.25",
        ]
        env = runtime_env(
            args.backend,
            args.tensorrt_lib_dir or [],
            args.openvino_dir,
            args.executorch_dir,
        )
        app_output = run_checked(infer_cmd, cwd=neuriplo_infer, timeout=args.infer_timeout, env=env)

        output_image = next(
            (path for path in output_candidates if path.is_file()),
            output_candidates[0],
        )
        require_file(output_image, "processed output image")
        stat = output_image.stat()
        if stat.st_size <= 0 or stat.st_mtime_ns < start_ns:
            raise RuntimeError(f"output image was not refreshed: {output_image} size={stat.st_size}")

        print(f"edgecrafter local e2e ok ({args.backend})")
        print(f"model repository: {model_repository}")
        print(f"model: {model}")
        print(f"output image: {output_image} ({stat.st_size} bytes)")
        if app_output.strip():
            print("app output tail:")
            print("\n".join(app_output.strip().splitlines()[-8:]))
        return 0

    runtime_bin = args.runtime_bin if args.runtime_bin is not None else default_runtime_bin(
        args.backend, args.transport
    )
    infer_bin = find_kserve_infer_binary(neuriplo_infer, args.infer_build_dir, args.infer_bin)
    output_image = neuriplo_infer / "data" / "output" / processed_image_name(args.task_type, "kserve")
    base_url = f"http://127.0.0.1:{args.port}"
    grpc_port = args.grpc_port if args.grpc_port is not None else args.port + 1
    if args.transport == "grpc":
        infer_endpoint = f"grpc://127.0.0.1:{grpc_port}"
    else:
        infer_endpoint = base_url

    tensorrt_lib_dirs = args.tensorrt_lib_dir or [
        Path("/home/oli/dependencies/TensorRT-10.13.3.9/targets/x86_64-linux-gnu/lib"),
        Path("/home/oli/dependencies/TensorRT-10.13.3.9/lib"),
    ]

    require_file(runtime_bin, f"neuriplo-kserve-runtime ({args.backend}) binary")
    require_file(infer_bin, "neuriplo-infer app binary")
    require_file(model, f"EdgeCrafter {args.backend} model artifact")
    require_file(args.source, "input image")
    require_file(args.labels, "labels file")

    log_path = Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"neuriplo-ecdet-e2e-{args.backend}-{args.transport}-{args.port}.log"
    )
    log_handle = log_path.open("w", encoding="utf-8")
    process: subprocess.Popen[str] | None = None

    try:
        runtime_cmd = [
            str(runtime_bin),
            "--model-name", args.model_name,
            "--model-path", str(model),
            "--backend", args.backend,
            "--port", str(args.port),
            "--instances", "1",
            "--request-timeout-ms", str(int(args.infer_timeout * 1000)),
        ]
        if args.transport == "grpc":
            runtime_cmd += ["--grpc-port", str(grpc_port)]
        process = subprocess.Popen(
            runtime_cmd,
            cwd=runtime_repo,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=runtime_env(
                args.backend,
                tensorrt_lib_dirs,
                args.openvino_dir,
                args.executorch_dir,
            ),
        )
        wait_until_ready(base_url, args.startup_timeout)

        metadata = http_json(f"{base_url}/v2/models/{args.model_name}", timeout=5.0)
        assert_edgecrafter_metadata(metadata, args.backend)

        infer_cmd = [
            str(infer_bin),
            f"--type={args.task_type}",
            f"--source={args.source}",
            f"--labels={args.labels}",
            f"--kserve_endpoint={infer_endpoint}",
            f"--kserve_model_name={args.model_name}",
            f"--kserve_transport={args.transport}",
            f"--kserve_timeout_ms={int(args.infer_timeout * 1000)}",
            "--min_confidence=0.25",
        ]
        infer_env = dict(os.environ)
        if args.transport == "http":
            infer_env["KSERVE_BINARY"] = "1"
        app_output = run_checked(
            infer_cmd, cwd=neuriplo_infer, timeout=args.infer_timeout, env=infer_env
        )

        require_file(output_image, "processed output image")
        stat = output_image.stat()
        if stat.st_size <= 0 or stat.st_mtime_ns < start_ns:
            raise RuntimeError(f"output image was not refreshed: {output_image} size={stat.st_size}")

        metrics = urllib.request.urlopen(f"{base_url}/metrics", timeout=5.0).read().decode("utf-8")
        model_label = f'{{model="{args.model_name}",version="1"}}'
        if args.transport == "http":
            if f'neuriplo_http_infer_requests_success_total{model_label} 1' not in metrics:
                raise RuntimeError("missing http success metric")
            if f'neuriplo_http_infer_requests_failure_total{model_label} 0' not in metrics:
                raise RuntimeError("http failure metric is not zero")
        else:
            if f'neuriplo_scheduler_requests_accepted_total{model_label} 1' not in metrics:
                raise RuntimeError("scheduler did not accept exactly one request")
            if f'neuriplo_scheduler_requests_rejected_total{model_label} 0' not in metrics:
                raise RuntimeError("scheduler rejected a request")
            if f'neuriplo_scheduler_requests_timed_out_total{model_label} 0' not in metrics:
                raise RuntimeError("scheduler timed out a request")

        print(f"edgecrafter kserve runtime e2e ok ({args.backend}, {args.transport})")
        print(f"runtime log: {log_path}")
        print(f"model repository: {model_repository}")
        print(f"model: {model}")
        print(f"output image: {output_image} ({stat.st_size} bytes)")
        if app_output.strip():
            print("app output tail:")
            print("\n".join(app_output.strip().splitlines()[-8:]))
        return 0
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        log_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKEND_DEFAULTS), default="onnx_runtime")
    parser.add_argument(
        "--mode",
        choices=("kserve", "local"),
        default="kserve",
        help="local runs in-process inference (openvino or executorch); kserve starts the runtime",
    )
    parser.add_argument(
        "--transport",
        choices=("http", "grpc"),
        default="http",
        help="KServe client transport (kserve mode only). grpc requires a runtime built with gRPC support.",
    )
    parser.add_argument("--port", type=int, default=19094, help="HTTP port (also used for readiness/metrics).")
    parser.add_argument("--grpc-port", type=int, default=None, help="gRPC port (default: --port + 1).")
    parser.add_argument(
        "--model-name",
        default=None,
        help="KServe model name (default: ecdet_s_<backend>, e.g. ecdet_s_executorch)",
    )
    parser.add_argument("--task-type", default="ecdet")
    parser.add_argument(
        "--model-repository",
        type=Path,
        default=DEFAULT_MODEL_REPOSITORY,
        help="Root model repository directory (default: $NEURIPLO_MODEL_REPOSITORY or ~/model_repository)",
    )
    parser.add_argument(
        "--prepare-model-repository",
        action="store_true",
        help="Run prepare_model_repository.py before the test when artifacts are missing",
    )
    parser.add_argument(
        "--skip-openvino-conversion",
        action="store_true",
        help="When preparing the repository, skip ovc IR conversion",
    )
    parser.add_argument(
        "--skip-executorch-conversion",
        action="store_true",
        help="When preparing the repository, skip ExecuTorch .pte export",
    )
    parser.add_argument("--runtime-bin", type=Path, default=None)
    parser.add_argument("--infer-build-dir", type=Path, default=None)
    parser.add_argument("--infer-bin", type=Path, default=None)
    parser.add_argument("--local-infer-build-dir", type=Path, default=None)
    parser.add_argument("--local-infer-bin", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None, help="Override model artifact path")
    parser.add_argument("--source", type=Path, default=REPOS_ROOT / "neuriplo-infer" / "data" / "dog.jpg")
    parser.add_argument("--labels", type=Path, default=REPOS_ROOT / "neuriplo-infer" / "labels" / "coco.names")
    parser.add_argument(
        "--tensorrt-lib-dir",
        type=Path,
        action="append",
        default=None,
        help="TensorRT lib directory added to LD_LIBRARY_PATH for the tensorrt backend (repeatable).",
    )
    parser.add_argument(
        "--openvino-dir",
        type=Path,
        default=default_openvino_root(),
        help="OpenVINO install root added to LD_LIBRARY_PATH for the openvino backend",
    )
    parser.add_argument(
        "--executorch-dir",
        type=Path,
        default=default_executorch_root(),
        help="ExecuTorch install root added to LD_LIBRARY_PATH for the executorch backend",
    )
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--infer-timeout", type=float, default=180.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "local" and args.backend not in LOCAL_BACKENDS:
        print(
            f"ERROR: --mode local requires --backend {' or '.join(sorted(LOCAL_BACKENDS))}",
            file=sys.stderr,
        )
        return 2

    if args.model_name is None:
        args.model_name = backend_model_name(args.backend)

    model = args.model if args.model is not None else resolve_model_artifact(args.backend, args.model_repository)
    if args.prepare_model_repository or not model.is_file():
        prepare_model_repository(
            args.backend,
            args.model_repository,
            args.skip_openvino_conversion,
            args.skip_executorch_conversion,
        )
        model = args.model if args.model is not None else resolve_model_artifact(args.backend, args.model_repository)

    try:
        return run_e2e(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
