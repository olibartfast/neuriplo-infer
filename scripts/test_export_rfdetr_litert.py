"""Regression checks for native LiteRT export and preset selection."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exporter", ROOT / "scripts/export_rfdetr_litert.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class NativeLiteRTTests(unittest.TestCase):
    def test_export_uses_native_format_and_copies_returned_artifact(self):
        for task, name in [("detection", "RFDETRMedium"), ("segmentation", "RFDETRSegMedium")]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                artifact = Path(tmp) / "upstream.tflite"
                artifact.write_bytes(b"native-model")
                calls = []
                class Model:
                    def export(self, **kwargs):
                        calls.append(kwargs)
                        return artifact
                module = types.SimpleNamespace(RFDETRMedium=Model, RFDETRSegMedium=Model)
                # Make the wrong family constructor fail, detecting task-routing regressions.
                wrong = "RFDETRSegMedium" if name == "RFDETRMedium" else "RFDETRMedium"
                setattr(module, wrong, lambda: self.fail("wrong model family"))
                with patch.dict(sys.modules, {"rfdetr": module}):
                    output = exporter.export_model(task, Path(tmp) / "deployed.tflite")
                self.assertEqual(output.read_bytes(), b"native-model")
                self.assertEqual(calls[0]["format"], "litert")
                self.assertEqual(calls[0]["output_name"], "deployed")

    def test_presets_select_native_export_for_litert_only(self):
        for preset, task in [("rfdetr", "detection"), ("rfdetrseg", "segmentation")]:
            for backend in ["litert", "onnxruntime"]:
                with self.subTest(preset=preset, backend=backend):
                    output = subprocess.check_output(
                        ["bash", str(ROOT / "docker_run_inference_e2e_example.sh"),
                         "--preset", preset, "--backend", backend, "--dry-run",
                         "--neuriplo-tasks-dir", "/tmp/tasks-placeholder"], text=True)
                    if backend == "litert":
                        self.assertIn("export_rfdetr_litert.py", output)
                        self.assertIn("--task " + task, output)
                        self.assertNotIn("onnx2tf", output)
                        self.assertIn(".tflite", output)
                    else:
                        self.assertNotIn("export_rfdetr_litert.py", output)
                        self.assertIn(".onnx", output)


if __name__ == "__main__":
    unittest.main()
