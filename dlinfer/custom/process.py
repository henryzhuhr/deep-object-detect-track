from typing import List
import numpy as np
import cv2


class Process:
    @staticmethod
    def preprocess(img: cv2.Mat, img_size=(640, 640)) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_t = cv2.resize(img, img_size)
        input_t = input_t.transpose(2, 0, 1)
        input_t = np.expand_dims(input_t, 0)
        input_t = input_t.astype(np.float32) / 255.0
        scale_h = img.shape[0] / input_t.shape[2]
        scale_w = img.shape[1] / input_t.shape[3]
        return input_t, scale_h, scale_w

    @staticmethod
    def postprocess(preds: np.ndarray, conf_thres=0.25, iou_thres=0.45) -> np.ndarray:
        preds = non_max_suppression(preds, conf_thres, iou_thres)[0]
        # print(preds.shape)
        # preds=cv2.dnn.NMSBoxesBatched(preds[:, :4], preds[:, 4], conf_thres, iou_thres)
        # exit()
        return preds

    @staticmethod
    def mark(img: cv2.Mat, preds: np.ndarray, label_list: List[str], scale_h=1.0, scale_w=1.0):
        for pred in preds:
            x1 = int(scale_w * pred[0])
            y1 = int(scale_h * pred[1])
            x2 = int(scale_w * pred[2])
            y2 = int(scale_h * pred[3])
            conf = pred[4] * 100
            cls = label_list[int(pred[5])]
            fontScale = 0.8
            color_RGB = (255, 0, 0)
            color = (color_RGB[2], color_RGB[1], color_RGB[0])
            thickness = 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                img,
                f"{conf:.1f}%:{cls}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                thickness,
            )


def non_max_suppression(
    predictions: np.ndarray,
    conf_thres=0.25,
    iou_thres=0.45,
    agnostic=False,
    max_det=300,
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(predictions, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        predictions = predictions[0]  # select only inference output

    bs = predictions.shape[0]  # batch size
    nc = predictions.shape[2] - 5  # number of classes
    xc = predictions[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    output = [np.zeros((0, 6), dtype=np.float32)] * bs
    for xi, x in enumerate(predictions):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x: np.ndarray = x[xc[xi]]  # confidence

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)

        # Detections matrix nx6 (xyxy, conf, cls)
        cls_preds = x[:, 5:]

        j: np.ndarray = np.argmax(cls_preds, 1)
        conf = np.zeros_like(j, dtype=np.float32)
        for i in range(cls_preds.shape[0]):
            conf[i] = cls_preds[i, j[i]]

        x = np.concatenate((box, conf[:, None], j[:, None]), axis=1)[conf > conf_thres]
        # x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        if x.shape[0] == 0:
            continue
        # 降序排列 从大到小
        x = x[x[:, 4].argsort()[::-1]][:max_nms]
        # x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        classes = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + classes, x[:, 4]  # boxes (offset by class), scores

        def non_max_suppression_impl(boxes: np.ndarray, confs: np.ndarray, iou_thres=0.6):
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = confs.flatten().argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= iou_thres)[0]
                order = order[inds + 1]
            # boxes = boxes[keep]
            # confs = confs[keep]
            return keep

        i = non_max_suppression_impl(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi] = x[i]

    return output


def xywh2xyxy(x: np.ndarray):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
