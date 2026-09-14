# Feature Requirements — <feature name>

> Copy this file to `specs/YYYY-MM-DD-<feature-name>/requirements.md` and delete
> the quoted guidance. Write this *before* `plan.md` and `validation.md`.

Roadmap phase: <link to the phase in ../roadmap.md>
Branch: `feature/<name>`

## Goal

What user- or system-visible outcome does this feature create? One paragraph.

## In Scope

- Observable behavior included in this branch. Name the flags, the routed task
  types, and the backends actually covered.

## Out of Scope

- Related work deliberately deferred, and where it goes instead (a roadmap phase,
  a sibling repo, an issue). "Helpful" expansion beyond this list is a review finding.

## Decisions

- Choice and its rationale. Record the ones where the obvious implementation would
  violate [`../tech-stack.md`](../tech-stack.md), where a sibling repo could
  plausibly have owned the change instead, or where a CLI contract moves.

## Constraints and Context

- Constitution rules that apply (boundaries, explicit non-choices).
- Existing patterns to reuse (the pipeline stage that owns this, the renderer
  path, the parser validation style).
- Cross-repo impact: does this need a `neuriplo-tasks` / `neuriplo` /
  `videocapture` / `neuriplo-kserve-client` change or pin move first?
- Contract surfaces touched: CLI flags, `--capabilities` output,
  `docs/capabilities.schema.json`, generated model-type docs.

## Open Questions

- Anything that could materially change the feature and has not been answered.
  Surface these before writing the plan, not after implementing.
