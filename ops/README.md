# Agentic Maintenance Control Plane

> Canonical home: `vision-platform/ops`. This directory is retained as a compatibility pointer for older workflows that still look inside `neuriplo-infer`. Cross-repository control-plane changes should be made in `vision-platform`, not here.

Historically, this directory defined control-plane assets for maintaining the
vision repo cluster while `neuriplo-infer` acted as the practical integration point.

The design is intentionally constrained:

- `neuriplo-tasks` owns task contracts, pre/postprocessing, and result types.
- `neuriplo` owns backend orchestration, backend adapters, and runtime/version compatibility.
- `videocapture` owns source handling and video backend behavior.
- `neuriplo-infer` owns the application layer, CLI, config, visualization, and
  end-to-end integration flow.

These files are meant to be consumed by agent runners, CI automation, or humans
reviewing agent-generated changes. They are not merge authority; `develop`
remains the integration branch and `master` remains release-only.

Current source of truth:

- `vision-platform/ops/CLUSTER_MAP.yaml`: cluster topology, ownership, and validation order
- `vision-platform/ops/policies.yaml`: allowed and forbidden automated change classes
- `vision-platform/ops/repo-meta/*.yaml`: repo-specific entrypoints, public surfaces, and constraints
- `vision-platform/ops/runbooks/`: execution guides for high-value maintenance flows
- `vision-platform/ops/PR_EVIDENCE_TEMPLATE.md`: standard evidence block for agent-generated PRs

The local files in this directory should not be extended. They can be removed after downstream automation has switched to `vision-platform/ops`.
