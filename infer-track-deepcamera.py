import imp
import os
import copy
import time
import argparse
from typing import Dict, List

import cv2
import numpy as np
import yaml
import pyrealsense2 as rs
import random

from dlinfer.detector import DetectorInferBackends
from dlinfer.detector import Process
from dlinfer.tracker import ByteTracker

class TTrackBbox:
    def __init__(self, tlwh: np.ndarray, objid: int, score: np.float32, clsid: int) -> None:
        self.tlwh = tlwh
        self.objid = objid
        self.score = score
        self.clsid = clsid

class TrackArgs:
    def __init__(self) -> None:
        args = self.get_args()

        self.model_path: str = args.model
        self.aspect_ratio_thresh: float = args.aspect_ratio_thresh
        self.min_box_area: float = args.min_box_area

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found")

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model", type=str, default=".cache/exp3/weights/best_openvino_model/best.xml")
        parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value.")
        parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
        return parser.parse_args()

def get_mid_pos(frame, box, depth_data, randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]  # 确定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))  # 确定深度搜索范围
    for i in range(randnum):
        bias = random.randint(-min_val // 4, min_val // 4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255, 0, 0), -1)
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波
    return np.mean(distance_list)

def plot_tracking(img: np.ndarray, tackbboxes: List[TTrackBbox], label_list: List[str], depth_data, display_info: str = ""):
    img = np.ascontiguousarray(np.copy(img))
    img_h, img_w = img.shape[:2]

    top_view = np.zeros([img_w, img_w, 3], dtype=np.uint8) + 255
    text_scale = 1.5
    text_thickness = 2
    line_thickness = 2
    radius = max(5, int(img_w / 140.0))

    cv2.putText(img, display_info, (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tackbbox in enumerate(tackbboxes):
        x1, y1, w, h = tackbbox.tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(tackbbox.objid)
        id_text = f"{label_list[tackbbox.clsid]}/{int(obj_id)} ({tackbbox.score:.2f})"
        color = get_color(abs(obj_id))
        cv2.rectangle(img, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(img, id_text, (intbox[0], intbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)

        # Get depth information
        dist = get_mid_pos(img, intbox, depth_data, 24)
        cv2.putText(img, f"{dist / 1000:.2f}m", (intbox[0], intbox[1] + 20), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 255, 255), thickness=text_thickness)

    return img

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def main():
    args = TrackArgs()

    # Initialize depth camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    # Load detector / 加载检测器
    backends = DetectorInferBackends()
    detector = backends.OpenVINOBackend(device="GPU.0")
    print("-- Available devices:", detector.query_device())
    detector.load_model(args.model_path, verbose=True)

    with open(".cache/yolov5/best_openvino_model/best.yaml", "r") as f:
        label_map: Dict[int, str] = yaml.load(f, Loader=yaml.FullLoader)["names"]
        label_list = list(label_map.values())
        print(label_list)

    tracker = ByteTracker()

    # (1, 3, 640, 640)
    dummy_inputs = np.random.randn(1, 3, 1280, 1280).astype(np.float32)
    output_t = detector.infer(dummy_inputs)

    frame_id = 0
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                start_time = time.time()
                img = color_image
                debug_img = copy.deepcopy(color_image)

                input_t, scale_h, scale_w = Process.preprocess(img,img_size=(1280,1280))
                output_t = detector.infer(input_t)
                preds = Process.postprocess(output_t)  # [ B, [x1, y1, x2, y2, conf, cls] ]
                online_targets = tracker.update(preds, scale_h, scale_w)
                online_tackbboxes: List[TTrackBbox] = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    tcls = t.clsid
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tackbboxes.append(TTrackBbox(tlwh, tid, t.score, tcls))

                display_info = f"Frame {frame_id}"
                frame_id += 1

                online_im = plot_tracking(
                    debug_img,
                    online_tackbboxes,
                    label_list,
                    depth_data=depth_image,
                    display_info=display_info,
                )

                # Display the resulting frame
                cv2.imshow('Inference', online_im)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                # Restart the pipeline if there's a runtime error
                pipeline.stop()
                pipeline.start(config)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
