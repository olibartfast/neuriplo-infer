# RF-DETR Pose Local vs KServe Integration Evidence

Validation status: Active compatibility set, evidence-tracked

Version set: `rfdetr-keypoint-pose-followup` from the pinned refs recorded in
`evidence.yaml` (sibling pins live in `versions.env` at the repo root).

Owning repos: `neuriplo-infer`, `neuriplo-tasks`, `neuriplo`, `neuriplo-kserve-client`, `neuriplo-kserve-runtime`, `videocapture`

## Purpose

Track compatibility evidence for the RF-DETR keypoint pose golden path across
local embedded inference and remote KServe gRPC inference.

## How evidence is produced

This scenario separates two classes of checks:

- `metadata CI`: run on any GitHub-hosted runner with no GPU or C++ tooling.
  Covers pinned-ref integrity (each sibling tag resolves to the commit SHA
  pinned in `versions.env`), version consistency, benchmark baseline
  schema, and result-contract drift.
- `attested`: require GPU/C++ tooling (configure, build, unit tests, local and
  KServe gRPC inference smoke). Routine CI does not run these. They are
  recorded as structured attestation in [`evidence.yaml`](evidence.yaml).

Regenerate the report:

```bash
scripts/generate_compat_report.py            # write reports/latest.md + .json
scripts/generate_compat_report.py --check    # CI guard: fail if stale or FAIL
scripts/generate_compat_report.py --offline  # skip ref reachability
```

The report is deterministic; CI fails if the committed copy does not match a
fresh regeneration. Add `--offline` only for local runs without network.

## Expected Checks

```text
clone pinned refs              (metadata CI)
validate result contracts      (metadata CI)
validate benchmark JSON        (metadata CI)
configure repositories         (attested)
build repositories             (attested)
run unit tests                 (attested)
run local embedded inference smoke test  (attested)
run KServe gRPC inference smoke test     (attested)
emit Markdown and JSON compatibility report
```

## Report Runner

`run.py` is an evidence-scaffold checker. It verifies that the baseline files
and report exist. The compatibility report itself is produced by
`scripts/generate_compat_report.py`, which stays in `neuriplo-platform`
(shared with the `kserve-ensemble` and `opencv-boundary` scenarios that remain
there). The inference benchmark commands remain in `neuriplo-infer`.

## Baselines

- `baselines/local.json`: baseline shape for `rfdetr-pose-local`
- `baselines/kserve-grpc.json`: baseline shape for `rfdetr-pose-kserve-grpc`

Null metric values mark measurements that have not been captured yet. The keys
remain present so downstream validation can rely on a stable artifact shape.
Once a measured run is published under `results/`, its metrics populate the
report summary.
