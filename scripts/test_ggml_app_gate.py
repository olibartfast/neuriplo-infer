#!/usr/bin/env python3
import argparse
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from ggml_app_gate import build_app_command, input_shape_without_batch, run_gate


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def make_args(root: Path, app: Path, converter: Path, **overrides) -> argparse.Namespace:
    values = {
        "onnx": root / "model.onnx",
        "source": root / "source.jpg",
        "app": app,
        "converter": converter,
        "output_dir": root / "out",
        "json_path": root / "out" / "evidence.json",
        "model_type": "yoloseg",
        "input_sizes": "1,3,640,640",
        "labels": None,
        "min_confidence": 0.4,
        "nms_threshold": 0.5,
        "mask_threshold": 0.5,
        "skip_convert": False,
        "timeout": 30,
        "python": sys.executable,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class GgmlAppGateTest(unittest.TestCase):
    def test_input_shape_without_batch(self):
        self.assertEqual(input_shape_without_batch("1,3,640,640"), "3,640,640")
        self.assertEqual(input_shape_without_batch("3,224,224"), "3,224,224")
        self.assertEqual(input_shape_without_batch("2,3,640,640"), "2,3,640,640")

    def test_app_command_carries_task_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "app"
            converter = root / "convert.py"
            args = make_args(root, app, converter)
            command = build_app_command(args, root / "model.gguf")
            self.assertIn("--type=yoloseg", command)
            self.assertIn("--input_sizes=1,3,640,640", command)
            self.assertIn("--min_confidence=0.4", command)
            self.assertIn("--nms_threshold=0.5", command)
            self.assertIn("--mask_threshold=0.5", command)
            self.assertIn("--no_display=true", command)

    def test_success_writes_evidence_and_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.onnx").write_bytes(b"onnx")
            (root / "source.jpg").write_bytes(b"jpg")
            converter = root / "convert.py"
            converter.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "out.write_bytes(b'GGUF')\n",
                encoding="utf-8",
            )
            app = root / "app.sh"
            write_executable(
                app,
                "#!/bin/sh\n"
                "mkdir -p data/output\n"
                "printf '{\"status\":\"success\"}' > data/output/run_report.json\n"
                "printf 'png' > data/output/processed_yoloseg_local.png\n",
            )
            args = make_args(root, app, converter)
            self.assertEqual(run_gate(args), 0)
            evidence = json.loads(args.json_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "ok")
            self.assertEqual(evidence["commands"]["convert"]["exit_code"], 0)
            self.assertEqual(evidence["commands"]["run"]["exit_code"], 0)
            self.assertEqual(len(evidence["renders"]), 1)
            self.assertEqual(evidence["run_report"]["status"], "success")
            self.assertEqual(evidence["gguf"]["bytes"], 4)

    def test_app_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.onnx").write_bytes(b"onnx")
            (root / "source.jpg").write_bytes(b"jpg")
            converter = root / "convert.py"
            converter.write_text(
                "import sys\nfrom pathlib import Path\n"
                "Path(sys.argv[sys.argv.index('--output') + 1]).write_bytes(b'GGUF')\n",
                encoding="utf-8",
            )
            app = root / "app.sh"
            write_executable(app, "#!/bin/sh\nexit 7\n")
            args = make_args(root, app, converter)
            self.assertEqual(run_gate(args), 1)
            evidence = json.loads(args.json_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed")
            self.assertEqual(evidence["commands"]["run"]["exit_code"], 7)

    def test_missing_input_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, root / "app", root / "convert.py")
            self.assertEqual(run_gate(args), 2)

    def test_converter_failure_writes_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.onnx").write_bytes(b"onnx")
            (root / "source.jpg").write_bytes(b"jpg")
            converter = root / "convert.py"
            converter.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
            app = root / "app.sh"
            write_executable(app, "#!/bin/sh\nexit 0\n")
            args = make_args(root, app, converter)
            self.assertEqual(run_gate(args), 1)
            evidence = json.loads(args.json_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed")
            self.assertEqual(evidence["commands"]["convert"]["exit_code"], 3)
            self.assertNotIn("run", evidence["commands"])

    def test_skip_convert_uses_existing_gguf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.onnx").write_bytes(b"onnx")
            (root / "source.jpg").write_bytes(b"jpg")
            out = root / "out"
            out.mkdir()
            (out / "model.gguf").write_bytes(b"GGUF")
            app = root / "app.sh"
            write_executable(
                app,
                "#!/bin/sh\n"
                "mkdir -p data/output\n"
                "printf '{\"status\":\"success\"}' > data/output/run_report.json\n",
            )
            args = make_args(
                root,
                app,
                root / "missing-converter.py",
                skip_convert=True,
                output_dir=out,
                json_path=out / "evidence.json",
            )
            self.assertEqual(run_gate(args), 0)
            evidence = json.loads(args.json_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "ok")
            self.assertTrue(evidence["commands"]["convert"]["skipped"])


if __name__ == "__main__":
    unittest.main()
