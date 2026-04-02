#!/usr/bin/env python3
"""Sync supported model/task types from vision-core README into this repo docs.

Single source of truth: the vision-core README.  This script propagates the
model-type list into every file that embeds it, using HTML marker comments.

Targets updated:
  1. docs/generated/supported-model-types.md  (full block, rewritten)
  2. README.md                                (SUPPORTED_MODEL_TYPES markers)
  3. .github/copilot-instructions.md          (MODEL_TYPES:COPILOT_INSTRUCTIONS markers)
  4. docs/ARCHITECTURE.md                     (MODEL_TYPES:ARCHITECTURE markers)

Validation (warnings, non-fatal):
  5. docs/TablePage.md — every extracted type string must appear somewhere in the file.

Usage:
  python scripts/sync_supported_model_types.py
  python scripts/sync_supported_model_types.py --vision-core-readme /path/to/vision-core/README.md
  python scripts/sync_supported_model_types.py --check   # dry-run, exit 1 if any file would change
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


# ---------------------------------------------------------------------------
# Marker definitions — each target file has its own named pair.
# ---------------------------------------------------------------------------

MARKERS: dict[str, tuple[str, str]] = {
    "readme": (
        "<!-- SUPPORTED_MODEL_TYPES:START -->",
        "<!-- SUPPORTED_MODEL_TYPES:END -->",
    ),
    "copilot": (
        "<!-- MODEL_TYPES:COPILOT_INSTRUCTIONS:START -->",
        "<!-- MODEL_TYPES:COPILOT_INSTRUCTIONS:END -->",
    ),
    "architecture": (
        "<!-- MODEL_TYPES:ARCHITECTURE:START -->",
        "<!-- MODEL_TYPES:ARCHITECTURE:END -->",
    ),
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_supported_block(vision_core_readme: str) -> str:
    """Return the markdown block under '### Supported Model Types (TaskFactory)'."""
    heading = "### Supported Model Types (TaskFactory)"
    start = vision_core_readme.find(heading)
    if start == -1:
        raise ValueError(f"Could not find heading: {heading}")

    after_heading = vision_core_readme.find("\n", start)
    if after_heading == -1:
        raise ValueError("Malformed README content after heading")
    after_heading += 1

    rest = vision_core_readme[after_heading:]
    match = re.search(r"^##\s+", rest, flags=re.MULTILINE)
    if not match:
        raise ValueError("Could not find next H2 section after supported model types block")

    block = rest[: match.start()].strip()
    if not block:
        raise ValueError("Extracted supported model types block is empty")
    return block


def extract_type_strings(block: str) -> list[str]:
    """Parse all quoted type strings (e.g. ``"yolo"``) out of the block."""
    return re.findall(r'"([a-z0-9_-]+)"', block)


# ---------------------------------------------------------------------------
# Category-summary builder (for ARCHITECTURE.md)
# ---------------------------------------------------------------------------

# Map from heading text in the block to the prose used in ARCHITECTURE.md.
_CATEGORY_MAP: list[tuple[str, str]] = [
    ("Object Detection", "Object Detection"),
    ("Open-Vocabulary Detection", "Open-Vocabulary Detection"),
    ("Instance Segmentation", "Instance Segmentation"),
    ("Classification", "Classification"),
    ("Video Classification", "Video Classification"),
    ("Optical Flow", "Optical Flow"),
    ("Pose Estimation", "Pose Estimation"),
    ("Depth Estimation", "Depth Estimation"),
]


def _extract_category_models(block: str, category: str) -> list[str]:
    """Return the model names listed under **<category>:** in the block."""
    pattern = rf"\*\*{re.escape(category)}:\*\*\s*\n((?:- .+\n?)+)"
    m = re.search(pattern, block)
    if not m:
        return []
    lines = m.group(1).strip().splitlines()
    models: list[str] = []
    for line in lines:
        # Pull the description after the last ` - ` on each bullet.
        desc_match = re.search(r"-\s+.*?\s+-\s+(.+)$", line)
        if desc_match:
            models.append(desc_match.group(1).strip())
    return models


def build_architecture_block(block: str) -> str:
    """Build the task-algorithm bullet list for ARCHITECTURE.md."""
    lines = ["- **Vision Task Algorithms**:"]
    for category, label in _CATEGORY_MAP:
        models = _extract_category_models(block, category)
        if models:
            lines.append(f"  - **{label}**: {', '.join(models)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Flat type list (for copilot-instructions.md)
# ---------------------------------------------------------------------------

def build_copilot_type_list(type_strings: list[str]) -> str:
    """Build a single-line backtick-quoted list of type strings."""
    formatted = ", ".join(f"`{t}`" for t in type_strings)
    return f"Valid values include: {formatted}."


# ---------------------------------------------------------------------------
# Generic marker replacement
# ---------------------------------------------------------------------------

def replace_between_markers(
    text: str,
    replacement: str,
    marker_start: str,
    marker_end: str,
) -> str:
    pattern = re.compile(
        rf"{re.escape(marker_start)}.*?{re.escape(marker_end)}",
        flags=re.DOTALL,
    )
    new_block = f"{marker_start}\n{replacement.strip()}\n{marker_end}"
    if not pattern.search(text):
        raise ValueError(f"Could not find markers {marker_start} … {marker_end}")
    return pattern.sub(new_block, text, count=1)


# ---------------------------------------------------------------------------
# TablePage validation
# ---------------------------------------------------------------------------

def validate_tablepage(table_text: str, type_strings: list[str]) -> list[str]:
    """Return warnings for type strings missing from TablePage.md."""
    warnings: list[str] = []
    for ts in type_strings:
        if ts not in table_text:
            warnings.append(f"  type string \"{ts}\" not found in docs/TablePage.md")
    return warnings


# ---------------------------------------------------------------------------
# File-write helper (supports --check dry-run)
# ---------------------------------------------------------------------------

def write_or_check(path: Path, new_content: str, *, check: bool) -> bool:
    """Write *new_content* to *path*.  Return True if the file changed.

    In check mode, do not write — only compare.
    """
    old_content = path.read_text(encoding="utf-8") if path.exists() else ""
    changed = old_content != new_content
    if changed and not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content, encoding="utf-8")
    return changed


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: exit 1 if any file would change (for CI)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_path = args.vision_core_readme or default_vision_core_readme(repo_root)

    if not source_path.exists():
        print(f"error: source README not found: {source_path}", file=sys.stderr)
        return 1

    source_text = source_path.read_text(encoding="utf-8")
    block = extract_supported_block(source_text)
    type_strings = extract_type_strings(block)

    if not type_strings:
        print("error: no type strings extracted from vision-core README", file=sys.stderr)
        return 1

    changed_files: list[str] = []

    # --- 1. docs/generated/supported-model-types.md (full rewrite) ----------
    generated_path = repo_root / "docs" / "generated" / "supported-model-types.md"
    source_label = "https://github.com/olibartfast/vision-core"
    generated_doc = (
        "# Supported Model Types\n\n"
        "Auto-generated from `vision-core` TaskFactory documentation.\n"
        "Do not edit manually; run `python scripts/sync_supported_model_types.py`.\n\n"
        f"Source: [{source_label}]({source_label})\n\n"
        f"{block}\n"
    )
    if write_or_check(generated_path, generated_doc, check=args.check):
        changed_files.append(str(generated_path.relative_to(repo_root)))

    # --- 2. README.md (SUPPORTED_MODEL_TYPES markers) ----------------------
    readme_path = repo_root / "README.md"
    if not readme_path.exists():
        print(f"error: {readme_path} not found", file=sys.stderr)
        return 1
    readme_text = readme_path.read_text(encoding="utf-8")
    readme_replacement = (
        f"{block}\n\n"
        "Canonical copy: [docs/generated/supported-model-types.md](docs/generated/supported-model-types.md)."
    )
    ms, me = MARKERS["readme"]
    updated_readme = replace_between_markers(readme_text, readme_replacement, ms, me)
    if write_or_check(readme_path, updated_readme, check=args.check):
        changed_files.append("README.md")

    # --- 3. .github/copilot-instructions.md (flat type list) ---------------
    copilot_path = repo_root / ".github" / "copilot-instructions.md"
    if copilot_path.exists():
        copilot_text = copilot_path.read_text(encoding="utf-8")
        copilot_replacement = build_copilot_type_list(type_strings)
        ms, me = MARKERS["copilot"]
        try:
            updated_copilot = replace_between_markers(copilot_text, copilot_replacement, ms, me)
            if write_or_check(copilot_path, updated_copilot, check=args.check):
                changed_files.append(".github/copilot-instructions.md")
        except ValueError as exc:
            print(f"warning: {copilot_path.name}: {exc}", file=sys.stderr)
    else:
        print(f"warning: {copilot_path} not found, skipping", file=sys.stderr)

    # --- 4. docs/ARCHITECTURE.md (task category summary) -------------------
    arch_path = repo_root / "docs" / "ARCHITECTURE.md"
    if arch_path.exists():
        arch_text = arch_path.read_text(encoding="utf-8")
        arch_replacement = build_architecture_block(block)
        ms, me = MARKERS["architecture"]
        try:
            updated_arch = replace_between_markers(arch_text, arch_replacement, ms, me)
            if write_or_check(arch_path, updated_arch, check=args.check):
                changed_files.append("docs/ARCHITECTURE.md")
        except ValueError as exc:
            print(f"warning: {arch_path.name}: {exc}", file=sys.stderr)
    else:
        print(f"warning: {arch_path} not found, skipping", file=sys.stderr)

    # --- 5. docs/TablePage.md (validation only) ----------------------------
    table_path = repo_root / "docs" / "TablePage.md"
    if table_path.exists():
        table_text = table_path.read_text(encoding="utf-8")
        warnings = validate_tablepage(table_text, type_strings)
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
    else:
        print(f"warning: {table_path} not found, skipping validation", file=sys.stderr)

    # --- Summary -----------------------------------------------------------
    if args.check:
        if changed_files:
            print("check failed — the following files are out of date:")
            for f in changed_files:
                print(f"  {f}")
            print("Run: python scripts/sync_supported_model_types.py")
            return 1
        print("check passed — all files are up to date")
        return 0

    if changed_files:
        print(f"Synced supported model types from: {source_path}")
        for f in changed_files:
            print(f"  Updated: {f}")
    else:
        print("All files already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
