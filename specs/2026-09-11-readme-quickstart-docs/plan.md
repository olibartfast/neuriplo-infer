# Feature Plan — README quickstart and user docs split

## Group 1 — Evidence before prose

1. Run the candidate quickstarts on Ubuntu 24.04 and keep the logs: `OPENCV_DNN`
   against current Ultralytics exports (expected to fail — decides the backend),
   the ONNX Runtime Docker image built from a clean `git archive`, and a native
   ONNX Runtime build. Commands that did not run do not go into the README.

## Group 2 — Generated inventory leaves the README

2. `scripts/sync_supported_model_types.py`: drop the `README.md` target; rewrite
   relative links in the upstream block to absolute neuriplo-tasks URLs.
3. Regenerate `docs/generated/supported-model-types.md`; confirm `--check` passes.
4. Update the rules that name the README block: `AGENTS.md`,
   `.cursor/rules/new-task-type-checklist.mdc`, `specs/tech-stack.md`.

## Group 3 — Sub-docs

5. `docs/Usage.md`: the README's CLI reference, examples, `--capabilities`, and
   run diagnostics, reconciled against `neuriplo-infer --help`.
6. `docs/KserveRuntime.md`: take over the KServe flag section and example;
   de-duplicate the environment variables it already tabulates.
7. `docs/Deployment.md`: platform support, backend matrix, Docker per backend,
   E2E presets, native builds, build options.

## Group 4 — README

8. Rewrite `README.md` around the verified quickstarts; keep `## Key Features`
   (a rule depends on it) and link every sub-doc.
9. `.dockerignore`: exclude `models/` and `environments/`.
10. Point `docs/DependencyManagement.md` platform notes at `Deployment.md`.

## Group 5 — Verification and integration

11. Execute `validation.md`; record results and deviations there.
12. `CHANGELOG.md` `[Unreleased]` entry; roadmap note.
