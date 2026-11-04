# TensorFlow classification re-gate

Maintainer authorized fixing TensorFlow ResNet-50 across neuriplo-tasks and this app on 2026-11-04.

R1: Validate the current upstream raw-BGR SavedModel contract against the current OpenVINO KNOWN-BAD classification status.
R2: Change coverage only on observed C++ correctness versus a Python reference.

Discovery: upstream commit 75cc7ac already embeds BGR mean subtraction and exports logits. Reuse it rather than changing generic task semantics again. No CLI/schema changes, releases, dependency pin moves, or other KNOWN-BAD fixes are in scope. Local sibling validation is not release-pin proof.
