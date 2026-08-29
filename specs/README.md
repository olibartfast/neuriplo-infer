# Specs

Spec-driven development for this repository, following the workflow in
[AI Spec-Driven Development](https://olibartfast.ninja/blog/ai-spec-driven-development-workflow.html):
durable intent lives in version-controlled files, not in a prompt. These specs
describe *this repo's own construction*, so they are addressed to whoever — human
or agent — changes it.

```
specs/
├── mission.md                 why this app exists, for whom, what "good" means
├── tech-stack.md              technical boundaries and explicit non-choices
├── roadmap.md                 the remaining delivery order, with status
├── architecture.md            ownership boundaries and runtime flow
├── procedures/                repeated mechanics, written once
│   └── merge-feature-branch.md
├── templates/                 copy these to start a feature packet
│   ├── requirements.md
│   ├── plan.md
│   └── validation.md
└── YYYY-MM-DD-<feature>/      one packet per roadmap phase, created when it starts
    ├── requirements.md        what
    ├── plan.md                how
    └── validation.md          proof
```

The constitution (`mission.md`, `tech-stack.md`, `roadmap.md`, `architecture.md`)
is deliberately short enough to read at the start of every planning task.

## The loop

1. **Pick a phase.** Take the next incomplete phase from
   [`roadmap.md`](roadmap.md); branch `feature/<name>` from `develop`; create
   `specs/YYYY-MM-DD-<feature-name>/`.
2. **Interview.** Resolve scope, decisions, and context with the maintainer
   before writing. An unanswered question that could materially change the
   feature gets asked, not assumed.
3. **Write the packet, in order:** `requirements.md`, then `plan.md`, then
   `validation.md` — from [`templates/`](templates/). Validation is written
   *before* implementation.
4. **Implement one task group at a time,** running the closest checks and
   inspecting the diff before moving on.
5. **Validate from `validation.md`,** executing it rather than recalling it.
   Record deviations honestly.
6. **Integrate.** Update `roadmap.md` and `CHANGELOG.md`, propagate any durable
   discovery into `tech-stack.md` or `architecture.md`, then merge spec and code
   together via [`procedures/merge-feature-branch.md`](procedures/merge-feature-branch.md).
7. **Replan.** Delivered work changes what should come next.

## Rules that keep this from becoming theater

- **When the requirement changes, update the spec in the same branch as the code.**
  Changing only the code makes the spec misleading; refusing the discovery makes
  it bureaucracy.
- **A discovery with project-wide reach updates the constitution,** not just the
  current packet — otherwise the newest feature follows one rule while older
  specs teach agents another.
- **Not every change needs a packet.** Use the full workflow when ambiguity, risk,
  handoff cost, or multi-step implementation justifies it. A typo fix, a link fix,
  or a one-line guard does not; a `CHANGELOG.md` entry is enough.
- **Specs are an interface, not a memory.** A different agent should be able to
  read this directory, run the checks, and continue the work.

## `specs/` versus `docs/`

`docs/` is for people using or building the project: runtime references,
compatibility matrices, export and dependency guides. Someone *running*
`neuriplo-infer` reads `docs/`. Someone *changing* it reads here.

Three things stay where they are, deliberately:

- **`AGENTS.md` and `CLAUDE.md` stay at the repository root.** Agents look for
  them there by convention; moving them would hide them. `AGENTS.md` remains the
  entry point, and points here.
- **`.cursor/rules/*.mdc` stay in `.cursor/`.** Cursor loads rules from that path.
- **`docs/capabilities.schema.json` stays in `docs/`.** It is a published machine
  contract that `--capabilities` output is validated against by the
  `capabilities_schema_contract` test and that consumers such as
  [neuriplo-ui](https://github.com/olibartfast/neuriplo-ui) read directly — a
  shipped artifact, not a plan for one.
