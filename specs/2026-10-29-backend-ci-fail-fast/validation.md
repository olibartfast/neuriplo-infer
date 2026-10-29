# Validation

Before implementation: save the original workflow and compare selected steps for each of the ten configurations after consolidation. Require all commands/actions and backend-specific conditions to remain present, unique step IDs, fail-fast true and no continue-on-error. Run actionlint and act dry-runs only; local full Docker CI would prune the shared daemon. Observe hosted failure cancellation before claiming the scheduling behavior validated. SDK download failures remain failures and must not be disguised as successful backend validation.

Local evidence: actionlint 1.7.12 passes. A transformation audit preserved all 10 configurations and 53 existing steps, including original conditions, command strings, cache separation, and timeouts. Hosted cancellation is pending.
