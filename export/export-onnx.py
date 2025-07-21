import os
from pathlib import Path

from ultralytics import YOLO

model_dir = Path(os.getenv("YOLO_MODEL_DIR", "~/.cache/ultralytics")) / "yolo11s.pt"

model = YOLO(model_dir)

path = model.export(format="onnx", dynamic=False, simplify=True)


def main():
    pass
