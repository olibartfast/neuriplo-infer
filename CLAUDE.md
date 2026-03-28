# Claude Instructions

- Treat `develop` as the integration branch for all normal work.
- Treat `master` as release-only. Pull requests targeting `master` should be reviewed for release readiness, not feature iteration.
- Prioritize correctness, regressions, missing tests, dependency safety, and public interface clarity.
- For C++ changes, focus on ownership, lifetime, thread safety, exception safety, and unnecessary copies.
- For inference changes, focus on shape and dtype assumptions, backend selection, device placement, latency regressions, and fallback behavior.
- Avoid style-only feedback unless it hides a correctness or maintenance problem.
