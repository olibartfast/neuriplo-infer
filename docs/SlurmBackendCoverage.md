# Slurm backend coverage checks

Use [`scripts/slurm_backend_matrix.py`](../scripts/slurm_backend_matrix.py) on
the existing Slurm controller to submit independent backend commands. The
controller client must match the cluster version. A local client failing with
`Unexpected message received` is not evidence of a backend failure.

Stage the runner, manifest, executables, libraries and model fixtures at identical
absolute paths on the chosen workers. Keep weights and tensors outside Git.
A manifest is a JSON list of cases:

```json
[
  {
    "name": "detector-executorch",
    "node": "cpu-worker-a",
    "command": ["/srv/coverage/executorch/neuriplo-infer", "--type=rfdetr", "--weights=/srv/coverage/models/detector.pte", "--source=/srv/coverage/dog.jpg", "--no_display=true", "--use-gpu=false"],
    "artifacts": ["/srv/coverage/executorch/neuriplo-infer", "/srv/coverage/models/detector.pte", "/srv/coverage/dog.jpg"],
    "env": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
    "timeout_seconds": 300,
    "memory_mib": 8192
  }
]
```

Include the backend shared library and tokenizer files in `artifacts` when
applicable. Commands are argument arrays, executed without a shell. Each case
runs in its own output directory. Paths in commands must be absolute.

```bash
python3 scripts/slurm_backend_matrix.py \
  --manifest /srv/coverage/cases.json --output /srv/coverage/results \
  --partition cpu --cpus 2 --minutes 20
squeue
scontrol show job <job-id>
```

Independent jobs are submitted without dependencies. Use workers with available
resources; if a cluster advertises placeholder memory, its scheduler may serialize
jobs on a node. The worker imposes its own address-space limit (8 GiB by default)
and kills the whole command process group on timeout. This limit includes mapped
model files and shared libraries; increase it explicitly for larger models.
No cluster configuration is changed.

`submitted.json` records every accepted submission. Each job retains
`runtime.log` and `result.json` with artifact SHA-256 hashes, node, UTC timestamps,
exit status and duration. Preserve `scontrol show job` output while completed
jobs remain in the controller's cache. Use `sacct` when accounting is enabled;
accounting is optional and was disabled in the 2026-11-04 cluster.

Exit zero is recorded as **RUNTIME_PASS**, not a numerical accuracy claim.
Compare tensors or inspect task results against the same-model reference before
updating [`BackendCoverage.md`](BackendCoverage.md). Preserve initial failures
alongside successful reruns, and identify local changes and SDK provenance
separately from released pins. See the active
[validation packet](../specs/2026-11-04-blocked-slurm-backends/validation.md).
