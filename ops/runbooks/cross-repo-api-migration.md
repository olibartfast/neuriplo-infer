# Cross-Repo API Migration Runbook

Use this runbook when a reviewed contract change in one repo requires mechanical
updates in a dependent repo.

## Goal

Propagate a reviewed source-repo contract change through downstream repos without
introducing silent behavior changes.

## Preconditions

- The source-repo contract change has already been reviewed by a human.
- The downstream work is mechanical and allowed by `ops/policies.yaml`.
- The affected dependency edge is declared in `ops/CLUSTER_MAP.yaml`.

## Procedure

1. Identify the source of truth.
   - `vision-core` for task and result contracts
   - `neuriplo` for backend orchestration and runtime contracts
   - `videocapture` for source and backend IO contracts

2. Enumerate downstream consumers from `ops/CLUSTER_MAP.yaml`.

3. Diff the old and new contract precisely.
   - symbol names
   - signatures
   - enums or string identifiers
   - config keys
   - expected output schema

4. Reject the task if it changes semantics rather than mechanical usage.
   - shape/dtype changes
   - fallback behavior changes
   - performance-critical runtime logic

5. Apply the minimal downstream update.
   - rename symbols
   - update call sites
   - update docs or generated metadata
   - add or update tests proving compatibility

6. Validate in order.
   - source repo tests if needed
   - downstream repo local tests
   - cross-repo integration check

7. Open linked PRs if more than one repo changes.
   - Each PR must be individually reviewable.
   - Each PR must explain the source-repo contract that triggered it.

## Exit Criteria

- all affected consumers updated
- no forbidden semantic drift
- compatibility evidence recorded
- PRs target `develop`
