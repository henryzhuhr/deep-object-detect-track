import os
import argparse
from typing import Dict, List

import cv2
from ultralytics import YOLO
import yaml

from dlinfer.detector.backend import DetectorInferBackends
from dlinfer.detector import Process

from ultralytics.nn.tasks import DetectionModel
from utils.dataset.types import BBOX_XYXY, ObjectLabel_BBOX_XYXY
from utils.dataset.variables import SUPPORTED_IMAGE_TYPES
import xml.etree.ElementTree as ET


class AutoLabelArgs:

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        # fmt: off
        parser.add_argument("-d", "--image-dir", type=str, default="~/data/drink.unlabel/cola")
        parser.add_argument("-c", "--dataset-config", type=str, default="~/data/drink-organized/dataset.yaml")
        parser.add_argument("-w", "--weight", type=str, default=".cache/yolov8/yolov8s.pt")
        parser.add_argument("-s", "--img-size", nargs="+", type=int, default=[640, 640])
        parser.add_argument("-t", "--conf-threshold", type=float, default=0.5)
        # fmt: on
        return parser.parse_args()

    def __init__(self) -> None:
        # fmt: off
        args = self.get_args()
        self.image_dir: str = os.path.expandvars(os.path.expanduser(args.image_dir))
        if not os.path.exists(self.image_dir): # check if the directory exists
            raise FileNotFoundError(f"Dataset directory not found: {self.image_dir}")
        self.dataset_config_file: str = os.path.expandvars(os.path.expanduser(args.dataset_config))
        if not os.path.exists(self.dataset_config_file): # check if the directory exists
            raise FileNotFoundError( f"Dataset configuration file not found: {self.dataset_config_file}")
        self.weight: str = args.weight
        if len(args.img_size) == 2:
            self.img_size: List[int] = args.img_size
        elif len(args.img_size) == 1:
            self.img_size: List[int] = args.img_size*2
        else:
            raise ValueError("Invalid img_size")
        self.conf_t: float = args.conf_threshold
        # fmt: on


def main():
    args = AutoLabelArgs()

    # =============== Load dataset configuration ===============
    with open(args.dataset_config_file, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    class_map: Dict[int, str] = data_config["names"]  # TODO: data type check
    print("-- class map :")
    for i, c in class_map.items():
        print(f"  {i}: {c}")

    # =============== Choose backend to Infer ===============
    model = DetectionModel("yolov8s.yaml", verbose=False)
    return
    # =======================================================

    image_dir = args.image_dir
    img_size = args.img_size

    for file in os.listdir(image_dir):
        # 获取文件后缀，查看是否是图片文件
        suffix = os.path.splitext(file)[-1]
        if suffix not in [f".{ext}" for ext in SUPPORTED_IMAGE_TYPES]:
            continue
        file_name = os.path.splitext(file)[0]
        xml_file = os.path.join(image_dir, f"{file_name}.xml")
        if os.path.exists(xml_file):
            print(f"File {xml_file} already exists. " f"If you want to re-label, please delete it by 'rm {xml_file}'")
            continue

        # =============== Detect ===============
        start_time = cv2.getTickCount()
        results = model.forward([os.path.join(image_dir, file)])
        result = results[0]
        end_time = cv2.getTickCount()
        infer_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
        print(f"Detection time: {infer_time:.2f} ms")
        # ======================================

        # ======================================


if __name__ == "__main__":
    main()
