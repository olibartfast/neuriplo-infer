# Feature Plan — <feature name>

> Copy to `specs/YYYY-MM-DD-<feature-name>/plan.md`. Derived from
> `requirements.md`. Numbered groups, dependency order, each ending in something
> observable. Specify behavior and consequential architecture; leave routine
> implementation to repository convention.

## Group 1 — Foundation

1. First independently verifiable task.
2. Second task.

## Group 2 — Behavior

3. Implement the user-visible slice (flag parsing and validation, pipeline
   wiring, routing, rendering — whichever this feature actually touches).

## Group 3 — Contract and documentation

4. Update the surfaces the change makes stale: `--capabilities`, the schema,
   `CHANGELOG.md`, `README.md`, the relevant `docs/` page. Regenerate — do not
   hand-edit — the supported-model-types block if model types moved.

## Group 4 — Verification

5. Add or update focused tests.
6. Run the checks declared in `validation.md`.

> Commit by task group for higher-risk work; keep the branch uncommitted until
> full validation for small features.
