import os
from pathlib import Path
from typing import Dict

from loguru import logger
from ultralytics import YOLO


def main():
    model_list = [
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
    ]
    model_record: Dict[str, bool] = dict.fromkeys(model_list, False)
    for m in model_list:
        try:
            download_export_model(m)
        except Exception as e:
            logger.error(f"Failed to download or export model {m}: {e}")
        else:
            model_record[m] = True


def download_export_model(model_name: str):
    model_dir = os.getenv("YOLO_MODEL_DIR", "~/.cache/ultralytics")
    model_path = Path(model_dir).expanduser().resolve() / model_name
    model = YOLO(model_path, verbose=True)
    path = model.export(format="onnx", dynamic=False, simplify=True)


if __name__ == "__main__":
    main()
