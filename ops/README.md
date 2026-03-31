# Agentic Maintenance Control Plane

This directory defines the first concrete control-plane assets for maintaining the
`vision-core`, `vision-inference`, `neuriplo`, and `videocapture` repo cluster.

The design is intentionally constrained:

- `vision-core` owns task contracts, pre/postprocessing, and result types.
- `neuriplo` owns backend orchestration, backend adapters, and runtime/version compatibility.
- `videocapture` owns source handling and video backend behavior.
- `vision-inference` owns the application layer, CLI, config, visualization, and
  end-to-end integration flow.

These files are meant to be consumed by agent runners, CI automation, or humans
reviewing agent-generated changes. They are not merge authority; `develop`
remains the integration branch and `master` remains release-only.

Contents:

- `CLUSTER_MAP.yaml`: cluster topology, ownership, and validation order
- `policies.yaml`: allowed and forbidden automated change classes
- `repo-meta/*.yaml`: repo-specific entrypoints, public surfaces, and constraints
- `runbooks/`: execution guides for high-value maintenance flows
- `PR_EVIDENCE_TEMPLATE.md`: standard evidence block for agent-generated PRs
