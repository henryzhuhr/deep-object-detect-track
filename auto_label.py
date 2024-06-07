import os
import argparse
from typing import Dict, List

import cv2
import yaml

from dlinfer.detector.backend import DetectorInferBackends
from dlinfer.detector import Process

from utils.dataset.types import BBOX_XYXY, ObjectLabel_BBOX_XYXY
from utils.dataset.variables import SUPPORTED_IMAGE_TYPES
import xml.etree.ElementTree as ET



class AutoLabelArgs:

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-d", "--image-dir", type=str, default="~/data/drink.unlabel/cola"
        )
        parser.add_argument(
            "-c",
            "--dataset-config",
            type=str,
            default="~/data/drink-organized/dataset.yaml",
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            # default=".cache/yolov5/yolov5s.onnx",
            default="temp/drink-yolov5x6/weights/best.onnx",
        )
        parser.add_argument("-t", "--conf-threshold", type=float, default=0.5)
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
        self.model: str = args.model
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
    backends = DetectorInferBackends()
    ## ------ ONNX ------
    # onnx_backend = backends.ONNXBackend
    # print("-- Available devices:", providers := onnx_backend.SUPPORTED_DEVICES)
    # detector = onnx_backend(
    #     device=providers, inputs=["images"], outputs=["output0"]
    # )

    ## ------ OpenVINO ------
    ov_backend = backends.OpenVINOBackend
    print("-- Available devices:", ov_backend.query_device())
    detector = ov_backend(device="AUTO")

    ## ------ TensorRT ------
    # detector = backends.TensorRTBackend()

    # =======================================================
    detector.load_model(args.model, verbose=True)

    image_dir = args.image_dir

    for file in os.listdir(image_dir):
        # 获取文件后缀，查看是否是图片文件
        suffix = os.path.splitext(file)[-1]
        if suffix not in [f".{ext}" for ext in SUPPORTED_IMAGE_TYPES]:
            continue
        file_name = os.path.splitext(file)[0]
        xml_file = os.path.join(image_dir, f"{file_name}.xml")
        if os.path.exists(xml_file):
            print(
                f"File {xml_file} already exists. "
                f"If you want to re-label, please delete it by 'rm {xml_file}'"
            )
            continue

        # =============== Auto label ===============
        start_time = cv2.getTickCount()
        img = cv2.imread(os.path.join(image_dir, file))  # H W C
        input_t, scale_h, scale_w = Process.preprocess(img)  # B C H W
        output_t = detector.infer(input_t)
        preds = Process.postprocess(output_t)
        end_time = cv2.getTickCount()
        infer_time = (end_time - start_time) / cv2.getTickFrequency() * 1000

        # print(f"File: {file}")
        # print(preds)

        bboxes: List[ObjectLabel_BBOX_XYXY] = []
        cls_cnt = 0
        for pred in preds:
            x1 = int(scale_w * pred[0])
            y1 = int(scale_h * pred[1])
            x2 = int(scale_w * pred[2])
            y2 = int(scale_h * pred[3])
            conf = pred[4]
            clsid = int(pred[5])
            if conf < args.conf_t:
                continue
            bbox = BBOX_XYXY(int(x1), int(y1), int(x2), int(y2))
            cls = class_map[clsid]
            bboxes.append(ObjectLabel_BBOX_XYXY(cls, bbox))
            cls_cnt += 1

        # =============== Save to xml ===============
        size = (img.shape[1], img.shape[0], img.shape[2])
        root = ET.Element("annotation")
        filename = ET.SubElement(root, "filename")
        filename.text = file
        size_node = ET.SubElement(root, "size")
        width = ET.SubElement(size_node, "width")
        width.text = str(size[0])
        height = ET.SubElement(size_node, "height")
        height.text = str(size[1])
        depth = ET.SubElement(size_node, "depth")
        depth.text = str(size[2])
        for obj in bboxes:
            object_node = ET.SubElement(root, "object")
            name = ET.SubElement(object_node, "name")
            name.text = obj.cls
            bndbox = ET.SubElement(object_node, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(obj.bbox.xmin)
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(obj.bbox.ymin)
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(obj.bbox.xmax)
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(obj.bbox.ymax)

        tree = ET.ElementTree(root)
        tree.write(xml_file, encoding="utf-8")
        print(
            f"Infer {infer_time:.3f} ms, File: {file}, {cls_cnt} objects saved to {xml_file} ({[b.cls for b in bboxes]})"
        )


if __name__ == "__main__":
    main()
