import argparse
import os
import torch
from ultralytics import YOLO

from aux import validate_file_exist


class DetectArgs:
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(add_help=False)
        # fmt: off
        parser.add_argument("-m", "--model", type=str, default="yolov8s")
        parser.add_argument("-p", "--pretrained-model", type=str, default=".cache/yolov8/yolov8s.pt")
        parser.add_argument("-c", "--dataset-config", type=str, default="configs/open-images-v7.yaml")
        parser.add_argument("-s", "--img-size", nargs="+", type=int, default=[640, 640])
        parser.add_argument("-i", "--input-image-or-dir", type=str, default="images/bus.jpg")
        parser.add_argument("-d", "--device", type=str, default="cuda", choices=["cpu", "cuda", "cuda:x", "mps"])
        # fmt: on
        return parser.parse_args()

    def __init__(self) -> None:
        args = self.get_args()
        self.model = args.model
        self.pretrained_model = validate_file_exist(args.pretrained_model)
        self.dataset_config = validate_file_exist(args.dataset_config)
        self.input_image_or_dir = validate_file_exist(args.input_image_or_dir)
        self.device = torch.device(args.device)


def main():
    args = DetectArgs()
    # Load a COCO-pretrained YOLOv8n model
    # model = YOLO(args.model)
    model = YOLO(args.pretrained_model)
    model.to(args.device)

    # Run batched inference on a list of images
    # https://docs.ultralytics.com/modes/predict/#inference-sources
    results = model([args.input_image_or_dir])  # return a list of Results objects

    os.makedirs("tmp", exist_ok=True)
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.save(filename="tmp/result.jpg")  # save to disk


if __name__ == "__main__":
    main()
