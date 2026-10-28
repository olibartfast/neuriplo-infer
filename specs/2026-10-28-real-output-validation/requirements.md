# Real-output validation

Before any release, validate the applicable task/model/backend matrix with real model weights and inspect actual predictions against references. A zero process exit code is runtime evidence only.

Provide an optional developer validation executable using the same application library, pipeline builder, task preprocessing/postprocessing and inference command as the CLI. Preserve raw input/output tensors, typed model metadata, structured predictions and rendered artifacts for reproducible comparison. The human-facing application remains unchanged.

Candidate execution requires verified remote source SHAs, clean rebuild provenance and exact executable hashes. Keep candidate evidence separate from published-release evidence. Record unsupported and blocked matrix entries explicitly.
