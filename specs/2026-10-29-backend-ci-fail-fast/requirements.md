# Backend CI fail-fast

When one backend build/test fails, cancel the other running or queued backend configurations in the same CI run. Preserve all six backend configurations and four KServe variants, their SDK pins, commands and test coverage. Use native matrix fail-fast without granting workflow cancellation permissions. Other independent workflows keep their existing behavior. No release or version bump.
