from typing import List, Tuple


class BBox_XYWH:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return f"[{self.x},{self.y},{self.w},{self.h}](xywh)"


class BBOX_XYXY:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __str__(self):
        return f"[{self.xmin},{self.ymin},{self.xmax},{self.ymax}](xyxy)"


class ObjectLabel_BBOX_XYXY:
    def __init__(self, cls: str, bbox: BBOX_XYXY):
        self.cls = cls
        self.bbox = bbox

    def __str__(self):
        return f"bbox:{self.cls}/{self.bbox}"


class ImageLabel:
    def __init__(
        self,
        parent_dir: str,
        image_file: str,
        label_file: str,
        size: Tuple[int, int, int],
        object_list: List[ObjectLabel_BBOX_XYXY],
    ):
        self.parent_dir = parent_dir
        self.image_file = image_file
        self.label_file = label_file
        self.size = size
        self.object_list = object_list

    def __str__(self):
        return f"[ {self.parent_dir}, {self.label_file} / {self.image_file} ]"

    def get_all_class(self) -> List[str]:
        return [obj.cls for obj in self.object_list]

    def to_voc(self):
        """
        将 ImageLabel 转换为 VOC 格式
        """
        raise NotImplementedError

    def norm_bbox_xyxy(self, bbox: BBOX_XYXY):
        """
        Convert bbox to normalized coordinates
        """
        dw = 1.0 / self.size[0]
        dh = 1.0 / self.size[1]
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        x1 = x1 * dw
        x2 = x2 * dw
        y1 = y1 * dh
        y2 = y2 * dh
        return BBOX_XYXY(x1, y1, x2, y2)

    def xyxy2xywh(self, bbox: BBOX_XYXY):
        """
        Convert bbox to normalized coordinates
        """
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return BBox_XYWH(x, y, w, h)

    def to_coco(self):
        """
        this coco is ultralytics format
        - https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#22-create-labels
        ## Returns

        """
        obj_list: List[List[int | float]] = []
        for obj in self.object_list:
            cls_name = obj.cls
            norm_bbox_xywh = self.xyxy2xywh(self.norm_bbox_xyxy(obj.bbox))
            x, y, w, h = norm_bbox_xywh.x, norm_bbox_xywh.y, norm_bbox_xywh.w, norm_bbox_xywh.h
            obj_list.append([cls_name, x, y, w, h])
        return obj_list
