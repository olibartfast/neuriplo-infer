#!/usr/bin/env python3
import tempfile
from pathlib import Path
import unittest
from ci_sdk_versions import resolve


class SDKVersionTest(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'versions.env'
            path.write_text(text)
            return resolve(path)

    def test_independent_versions_follow_metadata(self):
        for torch, ov in [('2.14.0', '2026.3.0'), ('3.1.2', '2027.4.5')]:
            self.assertEqual(self.read(f'PYTORCH_VERSION={torch}\nOPENVINO_VERSION={ov}\nONNX_RUNTIME_VERSION=1.28.0\n'),
                             {'pytorch_version': torch, 'openvino_version': ov, 'onnx_runtime_version': '1.28.0'})

    def test_missing_pin_fails(self):
        with self.assertRaises(KeyError):
            self.read('PYTORCH_VERSION=2.14.0\n')

    def test_invalid_or_duplicate_pin_fails(self):
        for text in ['PYTORCH_VERSION=master\nOPENVINO_VERSION=2026.3.0\n',
                     'PYTORCH_VERSION=2.14.0\nOPENVINO_VERSION=\n',
                     'PYTORCH_VERSION=2.14.0\nOPENVINO_VERSION=2026.3.0\nOPENVINO_VERSION=2025.2.0\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text)


if __name__ == '__main__':
    unittest.main()
