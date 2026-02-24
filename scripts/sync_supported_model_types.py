#!/usr/bin/env python3
"""Sync supported model/task types from vision-core README into this repo docs.

Usage:
  python scripts/sync_supported_model_types.py
  python scripts/sync_supported_model_types.py --vision-core-readme /path/to/vision-core/README.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


MARKER_START = "<!-- SUPPORTED_MODEL_TYPES:START -->"
MARKER_END = "<!-- SUPPORTED_MODEL_TYPES:END -->"


def extract_supported_block(vision_core_readme: str) -> str:
    heading = "### Supported Model Types (TaskFactory)"
    start = vision_core_readme.find(heading)
    if start == -1:
        raise ValueError(f"Could not find heading: {heading}")

    # Start after heading line.
    after_heading = vision_core_readme.find("\n", start)
    if after_heading == -1:
        raise ValueError("Malformed README content after heading")
    after_heading += 1

    # End at next H2 section.
    rest = vision_core_readme[after_heading:]
    match = re.search(r"^##\s+", rest, flags=re.MULTILINE)
    if not match:
        raise ValueError("Could not find next H2 section after supported model types block")

    block = rest[: match.start()].strip()
    if not block:
        raise ValueError("Extracted supported model types block is empty")
    return block


def replace_between_markers(text: str, replacement: str) -> str:
    pattern = re.compile(
        rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
        flags=re.DOTALL,
    )
    new_block = f"{MARKER_START}\n{replacement.strip()}\n{MARKER_END}"
    if not pattern.search(text):
        raise ValueError("Could not find marker block in README.md")
    return pattern.sub(new_block, text, count=1)


def default_vision_core_readme(repo_root: Path) -> Path:
    return repo_root / "build" / "_deps" / "vision-core-src" / "README.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync supported model types from vision-core README")
    parser.add_argument(
        "--vision-core-readme",
        type=Path,
        default=None,
        help="Path to vision-core README.md (default: build/_deps/vision-core-src/README.md)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    readme_path = repo_root / "README.md"
    generated_doc_path = repo_root / "docs" / "generated" / "supported-model-types.md"

    source_path = args.vision_core_readme or default_vision_core_readme(repo_root)
    if not source_path.exists():
        print(f"error: source README not found: {source_path}", file=sys.stderr)
        return 1
    if not readme_path.exists():
        print(f"error: README not found: {readme_path}", file=sys.stderr)
        return 1

    source_text = source_path.read_text(encoding="utf-8")
    project_readme = readme_path.read_text(encoding="utf-8")

    block = extract_supported_block(source_text)

    generated_doc = (
        "# Supported Model Types\n\n"
        "Auto-generated from `vision-core` TaskFactory documentation.\n"
        "Do not edit manually; run `python scripts/sync_supported_model_types.py`.\n\n"
        f"Source: `{source_path}`\n\n"
        f"{block}\n"
    )
    generated_doc_path.parent.mkdir(parents=True, exist_ok=True)
    generated_doc_path.write_text(generated_doc, encoding="utf-8")

    readme_replacement = (
        f"{block}\n\n"
        "Canonical copy: [docs/generated/supported-model-types.md](docs/generated/supported-model-types.md)."
    )
    updated_readme = replace_between_markers(project_readme, readme_replacement)
    readme_path.write_text(updated_readme, encoding="utf-8")

    print(f"Synced supported model types from: {source_path}")
    print(f"Updated: {readme_path}")
    print(f"Updated: {generated_doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
