import os
from typing import Dict

import cv2
import tqdm
import yaml

from modules.detector.process import Process
from modules.detector.tensort_detector_v10 import TensorRTDetectorV10

label_list = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
]


def main() -> None:
    detector = TensorRTDetectorV10()
    detector.load_model("~/.cache/ultralytics/yolo11n.engine", verbose=True)

    label_map: Dict[int, str] = {}
    with open("data/coco.yaml", "r") as f:
        d = yaml.safe_load(f)
        label_map = d["names"]
    label_list = [label_map[i] for i in range(len(label_map))]

    img_path = "images/bus.jpg"

    img = cv2.imread(img_path)

    os.makedirs("tmp", exist_ok=True)
    img_size = (640, 640)

    input_t, scale_h, scale_w = Process.preprocess(img, img_size)  # B C H W

    output_t = detector.infer(input_t)

    print("-- do inference")
    pbar = tqdm.tqdm(range(100))
    total_sum_time = 0
    _cnt = 0
    for i in pbar:
        start_time = cv2.getTickCount()
        # -- preprocess
        input_t, scale_h, scale_w = Process.preprocess(img, img_size)  # B C H W
        # -- inference
        output_t = detector.infer(input_t)
        end_time = cv2.getTickCount()
        infer_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
        # -- postprocess
        preds = Process.postprocess(output_t)
        Process.mark(img, preds, label_list, scale_h, scale_w)
        end_time = cv2.getTickCount()
        total_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
        total_sum_time += total_time
        _cnt += 1
        pbar.set_description(f"Time infer/total: {infer_time:.2f}/{total_time:.2f} ms")

    # -- mark
    print(f"-- Average time: {total_sum_time / _cnt:.2f} ms")
    cv2.imwrite(save_path := "tmp/out.jpg", img)
    print(f"-- output saved to '{save_path}'")


if __name__ == "__main__":
    main()
