#!/usr/bin/env python3
"""Dry-run evidence checker for the RF-DETR pose compatibility scenario."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = Path(__file__).resolve().parent

REQUIRED = [
    SCENARIO / "baselines" / "local.json",
    SCENARIO / "baselines" / "kserve-grpc.json",
    SCENARIO / "reports" / "latest.md",
]


def main() -> int:
    missing = [path for path in REQUIRED if not path.is_file()]
    if missing:
        for path in missing:
            print(f"ERROR: missing {path.relative_to(ROOT)}")
        return 1
    print("rfdetr pose local-vs-kserve evidence scaffold ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
