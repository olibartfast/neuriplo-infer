#!/usr/bin/env python3
"""Local smoke check for the app-owned E2E runner.

Validates that docker_run_inference_e2e_example.sh lists presets and can
dry-run a representative preset. Cluster-wide pin validation (sibling
checkouts against versions.env) is out of scope here; run it from the
checkout that owns the compatibility matrix.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def validate_runner(preset: str) -> None:
    runner = ROOT / "docker_run_inference_e2e_example.sh"
    if not runner.is_file():
        raise RuntimeError(f"missing app-owned E2E runner: {runner}")

    list_result = run(["bash", str(runner), "--list-presets"], ROOT)
    if list_result.returncode != 0:
        raise RuntimeError(f"failed to list presets\n{list_result.stdout}")
    if preset not in list_result.stdout.splitlines():
        raise RuntimeError(f"preset {preset!r} not listed by app-owned runner")

    dry_run = run(
        [
            "bash",
            str(runner),
            "--preset",
            preset,
            "--dry-run",
            "--neuriplo-tasks-dir",
            str(ROOT.parent / "neuriplo-tasks"),
            "--skip-export",
            "--skip-convert",
        ],
        ROOT,
    )
    if dry_run.returncode != 0:
        raise RuntimeError(f"dry-run failed for preset {preset}\n{dry_run.stdout}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="owlv2")
    args = parser.parse_args()

    try:
        validate_runner(args.preset)
    except Exception as exc:  # pragma: no cover - top-level CLI reporting
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("local inference smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
