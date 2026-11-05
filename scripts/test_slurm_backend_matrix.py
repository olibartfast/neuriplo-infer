#!/usr/bin/env python3
"""Exercise evidence retention for real worker processes and timeouts."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('matrix', Path(__file__).with_name('slurm_backend_matrix.py'))
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class WorkerTest(unittest.TestCase):
    def run_case(self, command, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        row = dict(name='test', node='test', command=command, artifacts=[sys.executable]) | overrides
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps([row]))
        code = matrix.execute(manifest, root / 'results', 0)
        directory = next((root / 'results').iterdir())
        return code, json.loads((directory / 'result.json').read_text()), directory

    def test_success_only_claims_runtime_pass(self):
        code, record, directory = self.run_case([sys.executable, '-c', 'print("real output")'])
        self.assertEqual(code, 0)
        self.assertEqual(record['status'], 'RUNTIME_PASS')
        self.assertIn(sys.executable, record['sha256'])
        self.assertIn('real output', (directory / 'runtime.log').read_text())

    def test_failure_preserves_exit_code(self):
        code, record, _ = self.run_case([sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual(code, 1)
        self.assertEqual(record['exit_code'], 7)

    def test_timeout_preserves_evidence(self):
        code, record, _ = self.run_case([sys.executable, '-c', 'import time; time.sleep(10)'], timeout_seconds=0.1)
        self.assertEqual(code, 1)
        self.assertEqual(record['exit_code'], 124)
        self.assertLess(record['duration_seconds'], 3)

    def test_missing_artifact_prevents_launch(self):
        code, record, _ = self.run_case([sys.executable, '-c', 'print("must not run")'], artifacts=['/does-not-exist'])
        self.assertEqual(code, 1)
        self.assertEqual(record['exit_code'], 125)
        self.assertIn('error', record)


if __name__ == '__main__':
    unittest.main()
