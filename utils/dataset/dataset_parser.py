import os
from typing import List

from numpy import object_

from utils.dataset.variables import SUPPORTED_IMAGE_TYPES
from .types import BBOX_XYXY, ImageLabel, ObjectLabel_BBOX_XYXY
import xml.etree.ElementTree as ET


class BaseDatasetParser:
    def __init__(self, label_file: str):
        self.label_file = os.path.expandvars(os.path.expanduser(label_file))

    @staticmethod
    def parse() -> ImageLabel:
        raise NotImplementedError

    # @staticmethod
    # def bbox_norm(il: ImageLabel) -> ImageLabel:
    #     size = il.size
    #     print(size)

    def __str__(self) -> str:
        raise NotImplementedError


class VOCParser(BaseDatasetParser):
    def __init__(self, label_file: str):
        super().__init__(label_file)
        self.data = self.parse(self.label_file)

    @staticmethod
    def parse(label_file: str) -> ImageLabel:
        parent_dir = os.path.dirname(label_file)
        with open(label_file, "r") as f:
            tree = ET.parse(label_file)
        root = tree.getroot()
        folder = root.find("folder").text
        filename = root.find("filename").text
        path = root.find("path").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        depth = int(size.find("depth").text)

        object_list: List[ObjectLabel_BBOX_XYXY] = []
        for object in root.iter("object"):
            cls = object.find("name").text
            bndbox = object.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            object_list.append(
                ObjectLabel_BBOX_XYXY(cls, BBOX_XYXY(xmin, ymin, xmax, ymax))
            )

        image_found = False

        for img_suffix in SUPPORTED_IMAGE_TYPES:
            image_file = f"{os.path.splitext(label_file)[0]}.{img_suffix}"
            if os.path.exists(image_file):
                image_found = True
                break
        if (not image_found) and (not os.path.exists(image_file)):
            raise FileNotFoundError(f"Image file not found for {image_file}")

        label_file = os.path.split(label_file)[-1]
        image_file = os.path.split(image_file)[-1]
        image_label = ImageLabel(
            parent_dir,
            image_file,
            label_file,
            (width, height, depth),
            object_list,
        )
        return image_label

    def __str__(self) -> str:
        return f"{self.data}"
