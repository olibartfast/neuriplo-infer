# Attempt log — Phase 2 release (0.12.0)

Scoreboard (packet 1, branch prep): `scripts/validate_release_pins.sh v0.12.0 && git diff --cached --stat && grep -q '^## \[0\.12\.0\] - 2026-11-12' CHANGELOG.md`

| Attempt | Config (role: tier) | Start SHA | Scoreboard cmd | Outcome | Interventions | Metrics | Notes |
|---|---|---|---|---|---|---|---|
| 0 | planner+implementer: single cloud model, one context | 2c21898..1d84f75 | n/a (pre-pyramid) | n/a — baseline, unmeasured | 1 (manual re-delegation after worker-type failure) | not captured | Pre-pyramid MCP landing (#111). Planner self-repaired after subagent harness error instead of re-delegating. Recorded as direction only, not evidence. |
| 1 | implementer: general subagent (fresh session) | 1d84f75 | n/a — stopped before work | STOPPED (clean-start gate) | 0 | no files touched, no branch, no commit | Planner fault: planner-owned scaffolding (`specs/attempt-log-phase-2-release.md`, untracked) dirtied the tree. Worker correctly aborted per packet precondition. Gate works as designed. |
| 2 | implementer: general subagent (fresh session; `worker` type broken in harness — `tool_choice` error) | 1d84f75 | `validate_release_pins.sh v0.12.0 && git show --stat HEAD && grep -n '^## ' CHANGELOG.md` | PASS | 0 | pins 4/4 ok; commit `02f1042`, 3 files, 4+/2-; full ctest NOT run on branch (deviation — release CI covers on PR) | Packet: `release/0.12.0` prep. Commit `02f1042` verified by planner diff-read + read-only reviewer (APPROVE). No push/tag/merge — planner-gated. Reviewer: general subagent, read-only, zero writes. |
