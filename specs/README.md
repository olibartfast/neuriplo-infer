# Specs

Agent-facing specifications for this repository: how it is meant to be
structured, and the procedures an agent is expected to follow. These describe
*this repo's own construction*, so they are addressed to whoever changes it.

- [`architecture.md`](architecture.md) — ownership boundaries, what this repo
  deliberately does not own, and the runtime flow through the app.
- [`feature-merge-roadmap.md`](feature-merge-roadmap.md) — the step-by-step
  GitFlow procedure for landing a `feature/*` branch on `develop`.

## What belongs here, and what does not

`docs/` is for people using or building the project: runtime references,
compatibility matrices, export and dependency guides, the detection
architecture guide. Someone running `neuriplo-infer` reads `docs/`. Someone
changing `neuriplo-infer` reads this directory.

Three things stay where they are, deliberately:

- **`AGENTS.md` and `CLAUDE.md` stay at the repository root.** Agents look for
  them there by convention; moving them would hide them. `AGENTS.md` remains the
  entry point, and points here.
- **`.cursor/rules/*.mdc` stay in `.cursor/`.** Cursor loads rules from that
  path.
- **`docs/capabilities.schema.json` stays in `docs/`.** It is a published
  machine contract that `--capabilities` output is validated against by the
  `capabilities_schema_contract` test and that consumers such as
  [neuriplo-ui](https://github.com/olibartfast/neuriplo-ui) read directly — a
  shipped artifact, not a plan for one.
