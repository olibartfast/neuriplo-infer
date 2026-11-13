#!/usr/bin/env python3
"""Dry-run/live harness for the remote-TPU KServe classification path.

--dry-run validates the scaffold files and prints the live command without
any network access or accelerator hardware. Routine CI runs only this mode
(CTest tpu_kserve_dry_run).

--live drives neuriplo-infer against a KServe V2 endpoint served by a TPU
runtime, checks the exit status and rendered artifact, and -- when both a
populated CPU baseline and a served top-5 file are available -- compares the
top-5 classes/scores within tolerance. The live path needs a Cloud TPU VM;
it never runs in routine CI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = Path(__file__).resolve().parent

BASELINE = SCENARIO / "baselines" / "cpu_top5.json"
EVIDENCE = SCENARIO / "evidence.yaml"
README = SCENARIO / "README.md"

MODEL_TYPE = "tensorflow-classifier"


def live_command(
    infer_bin: str,
    endpoint: str,
    model_name: str,
    transport: str,
    source: str,
    labels: str,
) -> list[str]:
    return [
        infer_bin,
        f"--type={MODEL_TYPE}",
        f"--source={source}",
        f"--labels={labels}",
        f"--kserve_endpoint={endpoint}",
        f"--kserve_model_name={model_name}",
        f"--kserve_transport={transport}",
    ]


def load_evidence(path: Path) -> dict[str, Any]:
    """Load the evidence file, using PyYAML when present.

    The fallback parser handles only the small nested-mapping subset used by
    evidence.yaml, so the dry-run check does not need an extra runtime
    dependency on CI images that ship without PyYAML.
    """
    text = path.read_text()
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return parse_evidence_subset(text)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("evidence.yaml did not parse to a mapping")
    return data


def parse_evidence_subset(text: str) -> dict[str, Any]:
    """Parse the indentation-based mapping subset used by evidence.yaml."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    block_scalar_indent: int | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if block_scalar_indent is not None:
            if indent > block_scalar_indent:
                continue
            block_scalar_indent = None
        stripped = raw.strip()
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        elif value == ">":
            parent[key] = ""
            block_scalar_indent = indent
        elif value in ("null", "~"):
            parent[key] = None
        elif value.lower() in ("true", "false"):
            parent[key] = value.lower() == "true"
        else:
            parent[key] = value.strip("\"'")
    return root


def check_dry_run() -> int:
    missing = [p for p in (README, BASELINE, EVIDENCE) if not p.is_file()]
    if missing:
        for path in missing:
            print(f"ERROR: missing {path.relative_to(ROOT)}")
        return 1
    try:
        baseline = json.loads(BASELINE.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: {BASELINE.relative_to(ROOT)} is not valid JSON: {exc}")
        return 1
    for key in ("source_image", "labels", "model_type", "classes", "scores",
                "score_tolerance"):
        if key not in baseline:
            print(f"ERROR: baseline missing key {key!r}")
            return 1
    try:
        evidence = load_evidence(EVIDENCE)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {EVIDENCE.relative_to(ROOT)} is not valid YAML: {exc}")
        return 1
    checks = evidence.get("checks")
    if not isinstance(checks, dict):
        print("ERROR: evidence.yaml is missing the 'checks' mapping")
        return 1
    for name in ("live_tpu_infer", "live_numeric_parity"):
        check = checks.get(name)
        if not isinstance(check, dict) or check.get("status") != "skipped":
            print(f"ERROR: evidence.yaml checks.{name}.status must be 'skipped'")
            return 1
    cmd = live_command(
        infer_bin="<build>/app/neuriplo-infer",
        endpoint="http://<tpu-host>:8000",
        model_name="<served-model>",
        transport="http",
        source="data/dog.jpg",
        labels="labels/imagenet_labels.txt",
    )
    required_flags = (
        f"--type={MODEL_TYPE}",
        "--kserve_endpoint=",
        "--kserve_model_name=",
        "--kserve_transport=",
    )
    for flag in required_flags:
        if not any(arg.startswith(flag) for arg in cmd):
            print(f"ERROR: generated live command is missing {flag}")
            return 1
    print("tpu-kserve evidence scaffold ok (dry-run; no network, no hardware)")
    print("live command would be:")
    print("  " + " ".join(cmd))
    return 0


def find_infer_binary(explicit: Path | None, build_dir: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates: list[Path] = []
    if build_dir is not None:
        candidates.append(build_dir / "app" / "neuriplo-infer")
    candidates.extend([
        ROOT / "build" / "app" / "neuriplo-infer",
        ROOT / "build-test" / "app" / "neuriplo-infer",
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "missing neuriplo-infer app binary; checked: "
        + ", ".join(str(p) for p in candidates)
    )


def compare_top5(baseline: dict[str, Any], served_path: Path) -> None:
    expected_classes = baseline.get("classes")
    expected_scores = baseline.get("scores")
    if expected_classes is None or expected_scores is None:
        print("numeric compare skipped: CPU baseline top-5 is not populated yet")
        return
    served = json.loads(served_path.read_text())
    got_classes = served.get("classes")
    got_scores = served.get("scores")
    if got_classes != expected_classes:
        raise RuntimeError(
            f"top-5 class mismatch: served {got_classes!r} "
            f"!= baseline {expected_classes!r}"
        )
    tolerance = float(baseline.get("score_tolerance", 1e-4))
    if len(got_scores) != len(expected_scores):
        raise RuntimeError("top-5 score count mismatch")
    for got, expected in zip(got_scores, expected_scores):
        if abs(float(got) - float(expected)) > tolerance:
            raise RuntimeError(
                f"top-5 score drift beyond {tolerance}: "
                f"served {got_scores!r} vs baseline {expected_scores!r}"
            )
    print(f"numeric compare ok: top-5 classes match, scores within {tolerance}")


def run_live(args: argparse.Namespace) -> int:
    infer_bin = find_infer_binary(args.infer_bin, args.infer_build_dir)
    for label, path in (("input image", args.source), ("labels file", args.labels)):
        if not path.is_file():
            print(f"ERROR: missing {label}: {path}")
            return 1
    output_dir = ROOT / "data" / "output"
    # Compare artifact mtimes against a start-time snapshot instead of file
    # names: the output name is deterministic, so a second live run overwrites
    # the same file and a name-based check would report it as "not refreshed".
    start_ns = time.time_ns()
    cmd = live_command(
        infer_bin=str(infer_bin),
        endpoint=args.endpoint,
        model_name=args.model_name,
        transport=args.transport,
        source=str(args.source),
        labels=str(args.labels),
    )
    print("+ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=args.infer_timeout)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: neuriplo-infer exited {result.returncode}")
        return 1
    refreshed = [
        p for p in output_dir.glob("processed_tensorflow-classifier_kserve*.png")
        if p.stat().st_size > 0 and p.stat().st_mtime_ns >= start_ns
    ]
    if not refreshed:
        print("ERROR: no refreshed processed_tensorflow-classifier_kserve*.png artifact")
        return 1
    artifact = refreshed[0]
    print(f"output artifact: {artifact} ({artifact.stat().st_size} bytes)")
    baseline = json.loads(BASELINE.read_text())
    if args.served_top5 is not None:
        try:
            compare_top5(baseline, args.served_top5)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1
    elif baseline.get("classes") is not None:
        print("numeric compare skipped: pass --served-top5 <json> "
              "with the served {classes, scores} to enforce it")
    else:
        print("numeric compare skipped: CPU baseline top-5 is not populated yet")
    report = {
        "endpoint": args.endpoint,
        "model_name": args.model_name,
        "transport": args.transport,
        "command": cmd,
        "exit_status": result.returncode,
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_bytes": artifact.stat().st_size,
        "numeric_compared": args.served_top5 is not None
        and baseline.get("classes") is not None,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"run report: {args.report_out}")
    print("tpu-kserve live ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate scaffold files; print live command only")
    parser.add_argument("--live", action="store_true",
                        help="run neuriplo-infer against a KServe V2 TPU endpoint")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model-name", default="tpu-classifier")
    parser.add_argument("--transport", default="http", choices=["http", "grpc"])
    parser.add_argument("--infer-bin", type=Path, default=None)
    parser.add_argument("--infer-build-dir", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "dog.jpg")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "labels" / "imagenet_labels.txt")
    parser.add_argument("--served-top5", type=Path, default=None,
                        help='JSON file {"classes": [...], "scores": [...]} '
                             "extracted from the served run for numeric compare")
    parser.add_argument("--report-out", type=Path,
                        default=SCENARIO / "reports" / "live-last.json")
    parser.add_argument("--infer-timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.live:
        try:
            return run_live(args)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    # Default (and --dry-run): scaffold check only, no network.
    return check_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
