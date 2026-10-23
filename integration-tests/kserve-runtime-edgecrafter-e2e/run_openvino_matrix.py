#!/usr/bin/env python3
"""Run EdgeCrafter ecdet OpenVINO validation: local, HTTP, and gRPC.

Each case reads model artifacts from the per-backend model repository folder
``${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_<backend>/1/`` (see ``prepare_model_repository.py``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "run.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=19094, help="HTTP port for the kserve cases")
    parser.add_argument("--prepare-model-repository", action="store_true")
    parser.add_argument("--skip-openvino-conversion", action="store_true")
    parser.add_argument("--model-repository", type=Path, default=None)
    args, extra = parser.parse_known_args()

    base_cmd = [sys.executable, str(RUNNER), "--backend", "openvino"]
    if args.model_repository is not None:
        base_cmd += ["--model-repository", str(args.model_repository)]
    if args.prepare_model_repository:
        base_cmd.append("--prepare-model-repository")
    if args.skip_openvino_conversion:
        base_cmd.append("--skip-openvino-conversion")
    base_cmd.extend(extra)

    cases = [
        ("local", base_cmd + ["--mode", "local"]),
        ("http", base_cmd + ["--mode", "kserve", "--transport", "http", "--port", str(args.port)]),
        (
            "grpc",
            base_cmd
            + [
                "--mode",
                "kserve",
                "--transport",
                "grpc",
                "--port",
                str(args.port + 10),
                "--grpc-port",
                str(args.port + 11),
            ],
        ),
    ]

    failures = 0
    for label, cmd in cases:
        print(f"=== openvino / {label} ===")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            print(f"FAILED: openvino / {label} (exit {result.returncode})", file=sys.stderr)
        else:
            print(f"PASSED: openvino / {label}")

    if failures:
        print(f"{failures} case(s) failed", file=sys.stderr)
        return 1
    print("openvino matrix ok (local, http, grpc)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
