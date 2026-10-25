"""EdgeCrafter ecdet Triton-style model repository layout.

Each backend gets its own model folder named ``<model>_<size>_<backend>``,
for example ``ecdet_s_executorch/1/model.pte``.
"""

from __future__ import annotations

from pathlib import Path

ECDET_MODEL_STEM = "ecdet_s"

BACKEND_REPO_SUFFIX: dict[str, str] = {
    "onnx_runtime": "onnx",
    "tensorrt": "tensorrt",
    "openvino": "openvino",
    "executorch": "executorch",
}


def backend_model_name(backend: str) -> str:
    try:
        suffix = BACKEND_REPO_SUFFIX[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported backend: {backend}") from exc
    return f"{ECDET_MODEL_STEM}_{suffix}"


def model_version_dir(model_repository: Path, backend: str) -> Path:
    return model_repository / backend_model_name(backend) / "1"
