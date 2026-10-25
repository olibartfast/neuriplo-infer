#!/usr/bin/env python3
"""Export EdgeCrafter ecdet to ExecuTorch .pte for local KServe E2E tests.

Uses the same PyTorch deploy wrapper as EdgeCrafter's ``export_onnx.py`` and
lowers with the XNNPACK delegate by default. Requires an EdgeCrafter checkout
(``edgecrafter-cpp-inference/third_party/EdgeCrafter`` by default) and the
matching ``ecdet_s.pth`` checkpoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPOS_ROOT = Path(__file__).resolve().parents[3]

EDGECRAFTER_ROOTS = [
    Path(os.environ.get("EDGECRAFTER_ROOT", "")),
    REPOS_ROOT / "edgecrafter-cpp-inference" / "third_party" / "EdgeCrafter",
    REPOS_ROOT / "EdgeCrafter",
]

CHECKPOINT_SOURCES = [
    REPOS_ROOT / "edgecrafter-cpp-inference" / "models" / "ecdet_s.pth",
    REPOS_ROOT / "EdgeCrafter" / "ecdetseg" / "ecdet_s.pth",
]


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_edgecrafter_root() -> Path:
    for candidate in EDGECRAFTER_ROOTS:
        if candidate.is_dir() and (candidate / "ecdetseg" / "tools" / "deployment" / "export_onnx.py").is_file():
            return candidate
    raise RuntimeError(
        "EdgeCrafter checkout not found. Clone it under edgecrafter-cpp-inference/third_party/EdgeCrafter "
        "or set EDGECRAFTER_ROOT."
    )


def resolve_python(edgecrafter_root: Path) -> Path:
    if "EDGECRAFTER_PYTHON" in os.environ:
        return Path(os.environ["EDGECRAFTER_PYTHON"])
    venv_python = edgecrafter_root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return venv_python
    return Path(sys.executable)


def export_pte(edgecrafter_root: Path, checkpoint: Path, output_path: Path, delegate: str) -> None:
    ecdetseg = edgecrafter_root / "ecdetseg"
    python = resolve_python(edgecrafter_root)
    inline = f"""
import os
import sys

sys.path.insert(0, {str(ecdetseg)!r})

import torch
import torch.nn as nn
from executorch.exir import to_edge, to_edge_transform_and_lower
from engine.core import YAMLConfig

cfg = YAMLConfig({str(ecdetseg / "configs/ecdet/ecdet_s.yml")!r}, resume={str(checkpoint)!r})
cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
state = torch.load({str(checkpoint)!r}, map_location="cpu")
state_dict = state["ema"]["module"] if "ema" in state else state["model"]
cfg.model.load_state_dict(state_dict)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)

model = Model().eval()
img_size = cfg.yaml_cfg["eval_spatial_size"]
images = torch.randn(1, 3, *img_size)
orig_target_sizes = torch.tensor([img_size], dtype=torch.int64)
with torch.no_grad():
    sample_outputs = model(images, orig_target_sizes)
    print("sample output shapes:", [tuple(item.shape) for item in sample_outputs])
exported_program = torch.export.export(model, (images, orig_target_sizes))

delegate = {delegate!r}
if delegate == "xnnpack":
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
else:
    executorch_program = to_edge(exported_program).to_executorch()

output_path = {str(output_path)!r}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as handle:
    handle.write(executorch_program.buffer)
print(f"exported ExecuTorch program: {{output_path}} (delegate={{delegate}})")
"""
    result = subprocess.run(
        [str(python), "-c", inline],
        cwd=ecdetseg,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "EdgeCrafter PyTorch ExecuTorch export failed:\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )
    if result.stdout.strip():
        print(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=None, help="Unused; kept for prepare_model_repository.py")
    parser.add_argument("--output", type=Path, required=True, help="Destination model.pte path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="EdgeCrafter checkpoint (.pth). Defaults to ecdet_s.pth from sibling repos.",
    )
    parser.add_argument(
        "--delegate",
        default=os.environ.get("NEURIPLO_EXECUTORCH_DELEGATE", "portable"),
        choices=("xnnpack", "portable"),
    )
    args = parser.parse_args()

    edgecrafter_root = resolve_edgecrafter_root()
    checkpoint = args.checkpoint if args.checkpoint is not None else first_existing(CHECKPOINT_SOURCES)
    if checkpoint is None or not checkpoint.is_file():
        raise RuntimeError(
            "missing ecdet_s.pth checkpoint; download per neuriplo-tasks/export/detection/edgecrafter/README.md "
            "or place it under edgecrafter-cpp-inference/models/ecdet_s.pth"
        )

    export_pte(edgecrafter_root, checkpoint, args.output, args.delegate)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
