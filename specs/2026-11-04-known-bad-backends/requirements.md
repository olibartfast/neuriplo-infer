# Remaining KNOWN-BAD backends

Maintainer requested all KNOWN-BAD fixes, extending the authorized TensorFlow work on 2026-11-04.

R1: Diagnose EdgeCrafter LiteRT with identical input bytes against ONNX and Python LiteRT before changing a backend.
R2: Provide correct supported RF-DETR detection/segmentation LiteRT export paths, preventing the known wrong generic conversion path from being the default recipe.
R3: Validate the current `rtdetrul` OpenVINO status and update its coverage entry.
R4: Validate local and KServe EdgeCrafter where prerequisites are available. Preserve bad-artifact warnings until a replacement passes.

No model weights committed, no releases, no kernel approximations, no weakening correctness criteria. Backend code belongs in neuriplo; task/export changes belong in neuriplo-tasks; presets and evidence belong here.
