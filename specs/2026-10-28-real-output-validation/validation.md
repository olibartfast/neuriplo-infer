# Validation

The candidate OpenVINO SDK build compiles the production library, CLI, optional model validator and app tests against the pinned candidate core/tasks commits. CTest reports 189 passed, 0 failed and 1 skipped (the optional Python JSON Schema dependency). Recorder tests preserve real-valued geometry and Float32 map bytes, retain INT64 values beyond Float64 exactness, cover every result variant, and reject mixed tensor types or overwriting existing evidence. Per-input metadata bridge regressions also pass.

Pending: clean candidate provenance for every SDK profile; trained-model numeric and visual comparisons through Chrome; the full applicable backend/model/mode matrix. Runtime success, semantic correctness, unavailable exports and hardware remain distinct. No release is authorized by these unit results.
