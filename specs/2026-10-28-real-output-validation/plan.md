# Plan

1. Add an opt-in validation executable that observes the existing task and renderer extension points.
2. Serialize all result variants and lossless tensor sidecars, retaining per-input dtype metadata.
3. Record this executable in the stack gate's clean-build provenance.
4. Exercise the candidate fixes with trained models in Chrome, then cover the full applicable matrix and compare numeric and visual outputs.
5. Add new regression tests for every confirmed defect; publish no release until validation is complete.
