#!/usr/bin/env python3
"""Run one GGML app-surface gate and write a JSON evidence record.

The script converts an ONNX artifact to the GGUF file the GGML loader reads,
runs `neuriplo-infer` with the requested task type, and records the exact
commands, exit codes, hashes, timings, app log, run report, and rendered
images. It deliberately does not decide PASS/BLOCKED: numeric comparison and
matrix promotion stay with the parity tests and the coverage evidence
section. Run it from the repository root, for example:

    python3 scripts/ggml_app_gate.py \
        --onnx /tmp/yolov8n-seg.onnx \
        --model-type yoloseg \
        --source data/dog.jpg \
        --input-sizes 1,3,640,640 \
        --labels labels/coco.names \
        --min-confidence 0.4 \
        --nms-threshold 0.5 \
        --mask-threshold 0.5

The default app is `build-ggml/app/neuriplo-infer`; the default converter is
the sibling `neuriplo/scripts/convert_onnx_to_gguf.py` next to this repo.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_shape_without_batch(input_sizes: str) -> str:
    """Convert `1,3,640,640` to the converter's `3,640,640` form."""
    parts = [part.strip() for part in input_sizes.split(",") if part.strip()]
    if len(parts) == 4 and parts[0] == "1":
        return ",".join(parts[1:])
    return ",".join(parts)


def git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_converter_command(args: argparse.Namespace, gguf: Path) -> list[str]:
    return [
        args.python,
        str(args.converter),
        "--input",
        str(args.onnx),
        "--output",
        str(gguf),
    ]


def build_app_command(args: argparse.Namespace, gguf: Path) -> list[str]:
    command = [
        str(args.app),
        f"--type={args.model_type}",
        f"--weights={gguf}",
        f"--source={args.source}",
        f"--input_sizes={args.input_sizes}",
        "--no_display=true",
        "--no_gif=true",
    ]
    if args.labels is not None:
        command.append(f"--labels={args.labels}")
    if args.min_confidence is not None:
        command.append(f"--min_confidence={args.min_confidence}")
    if args.nms_threshold is not None:
        command.append(f"--nms_threshold={args.nms_threshold}")
    if args.mask_threshold is not None:
        command.append(f"--mask_threshold={args.mask_threshold}")
    return command


def run_command(command: list[str], cwd: Path, log_path: Path, timeout: int) -> dict[str, Any]:
    started = _dt.datetime.now(_dt.timezone.utc)
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        timed_out = False
        output = result.stdout + result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as expired:
        timed_out = True
        output = (expired.stdout or "") + (expired.stderr or "")
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        exit_code = None
    duration_ms = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds() * 1000.0
    log_path.write_text(output, encoding="utf-8")
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_ms": round(duration_ms, 3),
        "log": str(log_path),
    }


def read_optional_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_gate(args: argparse.Namespace) -> int:
    args.onnx = args.onnx.resolve()
    args.source = args.source.resolve()
    args.app = args.app.resolve()
    args.converter = args.converter.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.labels is not None:
        args.labels = args.labels.resolve()

    for label, path in (("onnx", args.onnx), ("source", args.source), ("app", args.app)):
        if not path.is_file():
            print(f"error: {label} file not found: {path}", file=sys.stderr)
            return 2
    if not args.skip_convert and not args.converter.is_file():
        print(f"error: converter file not found: {args.converter}", file=sys.stderr)
        return 2
    if args.labels is not None and not args.labels.is_file():
        print(f"error: labels file not found: {args.labels}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gguf = args.output_dir / f"{args.onnx.stem}.gguf"
    app_cwd = args.output_dir / "run"
    app_cwd.mkdir(parents=True, exist_ok=True)

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model_type": args.model_type,
        "input_sizes": args.input_sizes,
        "onnx": {
            "path": str(args.onnx),
            "bytes": args.onnx.stat().st_size,
            "sha256": sha256_file(args.onnx),
        },
        "gguf": {"path": str(gguf)},
        "app": {
            "path": str(args.app),
            "bytes": args.app.stat().st_size,
            "sha256": sha256_file(args.app),
            "git_head": git_head(args.app.parent),
        },
        "sibling": {
            "converter": str(args.converter),
            "repo": str(args.converter.parent.parent),
            "git_head": git_head(args.converter.parent.parent),
        },
        "commands": {},
        "renders": [],
        "status": "failed",
    }

    if args.skip_convert and gguf.is_file():
        evidence["commands"]["convert"] = {"skipped": True, "output": str(gguf)}
    else:
        convert = run_command(
            build_converter_command(args, gguf),
            cwd=args.output_dir,
            log_path=args.output_dir / "convert.log",
            timeout=args.timeout,
        )
        evidence["commands"]["convert"] = convert
        if convert["exit_code"] != 0:
            evidence["gguf"] = {"path": str(gguf), "present": gguf.is_file()}
            _write_json(args.json_path, evidence)
            print(_summarize(evidence, args.json_path))
            return 1

    if not gguf.is_file():
        evidence["commands"]["convert"] = {
            "skipped": bool(args.skip_convert),
            "output": str(gguf),
            "error": "GGUF output does not exist",
        }
        _write_json(args.json_path, evidence)
        print(_summarize(evidence, args.json_path))
        return 1

    evidence["gguf"] = {
        "path": str(gguf),
        "bytes": gguf.stat().st_size,
        "sha256": sha256_file(gguf),
    }

    run = run_command(
        build_app_command(args, gguf),
        cwd=app_cwd,
        log_path=args.output_dir / "app.log",
        timeout=args.timeout,
    )
    evidence["commands"]["run"] = run
    evidence["run_report"] = read_optional_json(
        app_cwd / "data" / "output" / "run_report.json"
    )
    evidence["renders"] = sorted(
        str(path) for path in (app_cwd / "data" / "output").glob("*.png")
    )
    evidence["status"] = "ok" if run["exit_code"] == 0 else "failed"
    _write_json(args.json_path, evidence)
    print(_summarize(evidence, args.json_path))
    return 0 if evidence["status"] == "ok" else 1


def _write_json(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize(evidence: dict[str, Any], json_path: Path) -> str:
    run = evidence.get("commands", {}).get("run", {})
    return (
        f"status={evidence['status']} model_type={evidence['model_type']} "
        f"run_exit={run.get('exit_code')} renders={len(evidence.get('renders', []))} "
        f"evidence={json_path}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="input ONNX artifact")
    parser.add_argument("--model-type", required=True, help="neuriplo-infer --type value")
    parser.add_argument("--source", required=True, type=Path, help="image or video source")
    parser.add_argument(
        "--input-sizes",
        required=True,
        help="app --input_sizes value, e.g. 1,3,640,640",
    )
    parser.add_argument(
        "--app",
        type=Path,
        default=Path("build-ggml/app/neuriplo-infer"),
        help="neuriplo-infer binary (default: build-ggml/app/neuriplo-infer)",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "neuriplo"
        / "scripts"
        / "convert_onnx_to_gguf.py",
        help="sibling GGML converter (default: ../neuriplo/scripts/convert_onnx_to_gguf.py)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for the converter (default: this interpreter)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build-ggml-gate"),
        help="evidence/output directory (default: build-ggml-gate)",
    )
    parser.add_argument("--json", dest="json_path", type=Path, help="evidence JSON path")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--mask-threshold", type=float)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--timeout", type=int, default=600, help="per-command timeout seconds")
    args = parser.parse_args(argv)
    if args.json_path is None:
        args.json_path = args.output_dir / "evidence.json"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.json_path = args.json_path.resolve()
    return run_gate(args)


if __name__ == "__main__":
    sys.exit(main())
