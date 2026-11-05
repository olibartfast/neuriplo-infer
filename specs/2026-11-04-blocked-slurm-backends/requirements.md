# Blocked backend cells

Maintainer request: resolve BLOCKED cells in BackendCoverage.md and test independent backends concurrently through Slurm. Continue the existing coverage feature; preserve the previously validated LiteRT changes.

R1: Reproduce OpenCV DNN multi-input and RF-DETR ExecuTorch blockers with exact artifacts.
R2: Apply mechanical adapter/build/export fixes in the owning repository; preserve model mathematics, CLI, outputs, device selection and release pins. Runtime upgrades and new operators are not assumed authorized.
R3: Submit independent tests through the existing Slurm controller with resource limits, isolated output directories and recorded job/accounting evidence.
R4: Update cells only after numerical/reference or meaningful rendered-output checks; retain real upstream blockers and document them. Include prose blockers such as KServe gRPC and TorchScript in the inventory.
