# Claude Instructions

- This repo is part of the `vision-stack` cluster. A sibling application repo, [vision-tracking](https://github.com/olibartfast/vision-tracking), handles detection + tracking pipelines using the same shared libraries independently.
- Consult `docs/Roadmap.md` for project direction and priorities.
- Treat `develop` as the integration branch for all normal work.
- Push normal feature, fix, refactor, docs, and chore work on a short-lived topic branch and open the pull request into `develop`.
- Do not treat direct pushes to `master` as part of the normal workflow.
- Treat `master` as release-only. Pull requests targeting `master` should be reviewed for release readiness, not feature iteration.
- Before creating a release, update both `VERSION` and `CHANGELOG.md`.
- After finishing a release, delete any temporary branches created ad hoc for that release.
- Prioritize correctness, regressions, missing tests, dependency safety, and public interface clarity.
- For C++ changes, focus on ownership, lifetime, thread safety, exception safety, and unnecessary copies.
- For inference changes, focus on shape and dtype assumptions, backend selection, device placement, latency regressions, and fallback behavior.
- Avoid style-only feedback unless it hides a correctness or maintenance problem.
