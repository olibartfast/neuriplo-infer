"""Export RF-DETR through its native LiteRT path (never generic ONNX conversion)."""
import argparse
from pathlib import Path
import shutil


def export_model(task, output):
    from rfdetr import RFDETRMedium, RFDETRSegMedium

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = RFDETRMedium() if task == "detection" else RFDETRSegMedium()
    artifact = Path(model.export(format="litert", output_dir=str(output.parent),
                                 output_name=output.stem))
    if artifact.resolve() != output.resolve():
        shutil.copyfile(artifact, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["detection", "segmentation"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_model(args.task, args.output)
