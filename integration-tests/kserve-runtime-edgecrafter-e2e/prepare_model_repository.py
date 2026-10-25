#!/usr/bin/env python3
"""Prepare the local Triton-style model repository layout for EdgeCrafter ecdet.

Each backend has its own model folder named ``<model>_<size>_<backend>``:

  ${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_onnx/1/model.onnx
  ${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_tensorrt/1/model.engine
  ${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_openvino/1/model.{xml,bin}
  ${NEURIPLO_MODEL_REPOSITORY}/ecdet_s_executorch/1/model.pte

Run once before the e2e runners when artifacts are missing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repository_layout import BACKEND_REPO_SUFFIX, backend_model_name, model_version_dir

REPOS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO = Path(os.environ.get("NEURIPLO_MODEL_REPOSITORY", Path.home() / "model_repository"))

# Triton-style platform label per backend. The neuriplo runtime advertises its
# own ``neuriplo_<backend>`` platform; this field is for repository/Triton
# compatibility and human readability only.
BACKEND_PLATFORM: dict[str, str] = {
    "onnx_runtime": "onnxruntime_onnx",
    "tensorrt": "tensorrt_plan",
    "openvino": "openvino",
    "executorch": "executorch_pte",
}

# Backends whose model format does not carry tensor names, so the server cannot
# auto-complete metadata from the model and needs an explicit config.pbtxt. ONNX,
# TensorRT, and OpenVINO all self-describe (Triton-style auto-complete), so they
# get no generated config and report their own real names/datatypes.
CONFIG_REQUIRED_BACKENDS: set[str] = {"executorch"}

# Authoritative EdgeCrafter ecdet dual-input contract for name-less backends.
# The runtime overlays these I/O names and datatypes onto backend metadata, so a
# name-less model format (ExecuTorch .pte) advertises the right contract without
# any model-specific hardcoding in the backend itself.
ECDET_CONFIG_BODY = """\
max_batch_size: 0

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 1, 3, 640, 640 ]
  },
  {
    name: "orig_target_sizes"
    data_type: TYPE_INT64
    dims: [ 1, 2 ]
  }
]

output [
  {
    name: "labels"
    data_type: TYPE_INT64
    dims: [ 1, 300 ]
  },
  {
    name: "boxes"
    data_type: TYPE_FP32
    dims: [ 1, 300, 4 ]
  },
  {
    name: "scores"
    data_type: TYPE_FP32
    dims: [ 1, 300 ]
  }
]
"""


def ensure_config_pbtxt(model_repository: Path, backend: str) -> None:
    dest = model_repository / backend_model_name(backend) / "config.pbtxt"
    if dest.is_file():
        print(f"ok: config.pbtxt already present at {dest}")
        return
    platform = BACKEND_PLATFORM.get(backend, "neuriplo_" + backend)
    contents = f'name: "{backend_model_name(backend)}"\nplatform: "{platform}"\n{ECDET_CONFIG_BODY}'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(contents)
    print(f"wrote config.pbtxt: {dest}")

ONNX_SOURCES = [
    DEFAULT_REPO / "ecdet_s_onnx" / "1" / "model.onnx",
    DEFAULT_REPO / "ecdet" / "1" / "model.onnx",
    REPOS_ROOT / "edgecrafter-cpp-inference" / "models" / "ecdet_s.onnx",
    REPOS_ROOT / "neuriplo-infer" / "models" / "e2e" / "ecdet_s.onnx",
]

TRT_SOURCES = [
    DEFAULT_REPO / "ecdet_s_tensorrt" / "1" / "model.engine",
    DEFAULT_REPO / "ecdet" / "1" / "model.engine",
    REPOS_ROOT / "edgecrafter-cpp-inference" / "models" / "ecdet_s.trt.engine",
]

PTE_SOURCES = [
    DEFAULT_REPO / "ecdet_s_executorch" / "1" / "model.pte",
    DEFAULT_REPO / "ecdet" / "1" / "model.pte",
    REPOS_ROOT / "edgecrafter-cpp-inference" / "models" / "ecdet_s.pte",
    REPOS_ROOT / "neuriplo-infer" / "models" / "e2e" / "ecdet_s.pte",
]


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def copy_if_missing(source: Path, dest: Path, label: str) -> None:
    if dest.is_file():
        print(f"ok: {label} already present at {dest}")
        return
    if not source.is_file():
        raise RuntimeError(f"missing source for {label}: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    print(f"copied {label}: {source} -> {dest}")


def convert_openvino_ir(onnx_path: Path, xml_path: Path) -> None:
    if xml_path.is_file() and xml_path.with_suffix(".bin").is_file():
        print(f"ok: OpenVINO IR already present at {xml_path}")
        return
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = xml_path.with_suffix("")
    cmd = [
        "ovc",
        str(onnx_path),
        "--output_model",
        str(output_stem),
        "--input",
        "images[1,3,640,640]",
        "--input",
        "orig_target_sizes[1,2]",
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ovc failed (install openvino-dev or set PATH):\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )
    if not xml_path.is_file():
        raise RuntimeError(f"ovc did not produce {xml_path}")


def export_executorch_pte(pte_path: Path) -> None:
    if pte_path.is_file():
        print(f"ok: ExecuTorch program already present at {pte_path}")
        return
    source = first_existing(PTE_SOURCES)
    if source is not None and source != pte_path:
        copy_if_missing(source, pte_path, "ExecuTorch program")
        return
    pte_path.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "export_ecdet_executorch.py"
    cmd = [
        sys.executable,
        str(script),
        "--output",
        str(pte_path),
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ExecuTorch export failed (install EdgeCrafter deps + executorch in EdgeCrafter .venv, "
            "or place model.pte manually):\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )
    if not pte_path.is_file():
        raise RuntimeError(f"export did not produce {pte_path}")


def prepare_onnx(model_repository: Path) -> Path:
    dest = model_version_dir(model_repository, "onnx_runtime") / "model.onnx"
    if dest.is_file():
        print(f"ok: ONNX already present at {dest}")
        return dest
    source = first_existing(ONNX_SOURCES)
    if source is None:
        raise RuntimeError(
            "no ecdet ONNX found; export per neuriplo-tasks/export/detection/edgecrafter/README.md "
            f"or place model.onnx under {dest.parent}/"
        )
    copy_if_missing(source, dest, "ONNX")
    return dest


def prepare_tensorrt(model_repository: Path) -> None:
    dest = model_version_dir(model_repository, "tensorrt") / "model.engine"
    if dest.is_file():
        print(f"ok: TensorRT engine already present at {dest}")
        return
    source = first_existing(TRT_SOURCES)
    if source is None:
        print(f"warn: no TensorRT engine source found; {dest} still missing")
        return
    copy_if_missing(source, dest, "TensorRT engine")


def prepare_openvino(model_repository: Path, onnx_path: Path) -> None:
    xml_path = model_version_dir(model_repository, "openvino") / "model.xml"
    convert_openvino_ir(onnx_path, xml_path)


def prepare_executorch(model_repository: Path) -> None:
    pte_path = model_version_dir(model_repository, "executorch") / "model.pte"
    export_executorch_pte(pte_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-repository",
        type=Path,
        default=DEFAULT_REPO,
        help="Root model repository directory (default: $NEURIPLO_MODEL_REPOSITORY or ~/model_repository)",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_REPO_SUFFIX),
        action="append",
        help="Prepare only the selected backend folder(s). Default: all backends.",
    )
    parser.add_argument("--skip-openvino", action="store_true", help="Do not run ovc IR conversion")
    parser.add_argument("--skip-executorch", action="store_true", help="Do not export/copy ExecuTorch model.pte")
    args = parser.parse_args()

    backends = set(args.backend or BACKEND_REPO_SUFFIX)
    model_repository = args.model_repository

    onnx_path: Path | None = None
    if "onnx_runtime" in backends or "openvino" in backends:
        onnx_path = prepare_onnx(model_repository)
    if "tensorrt" in backends:
        prepare_tensorrt(model_repository)
    if "openvino" in backends and not args.skip_openvino:
        if onnx_path is None:
            onnx_path = prepare_onnx(model_repository)
        prepare_openvino(model_repository, onnx_path)
    if "executorch" in backends and not args.skip_executorch:
        prepare_executorch(model_repository)

    for backend in sorted(backends):
        if backend not in CONFIG_REQUIRED_BACKENDS:
            continue
        if backend == "executorch" and args.skip_executorch:
            continue
        ensure_config_pbtxt(model_repository, backend)

    prepared = [backend_model_name(backend) for backend in sorted(backends)]
    print(f"model repository ready for: {', '.join(prepared)} under {model_repository}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
