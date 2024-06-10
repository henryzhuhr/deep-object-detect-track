import os
import copy
import time
import argparse
from typing import Dict, List

import cv2
import numpy as np
import yaml

from dlinfer.detector import DetectorInferBackends
from dlinfer.detector import Process
from dlinfer.tracker import ByteTracker


class TTrackBbox:
    def __init__(
        self, tlwh: np.ndarray, objid: int, score: np.float32, clsid: int
    ) -> None:
        self.tlwh = tlwh
        self.objid = objid
        self.score = score
        self.clsid = clsid


class TrackArgs:

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        # fmt: off
        parser.add_argument("-v", "--video", type=str, default=".cache/palace.mp4")
        parser.add_argument("-o", "--outdir", type=str, default="tmp")
        parser.add_argument("-m", "--model", type=str, default=".cache/yolov5/yolov5s_openvino_model/yolov5s.xml")
        parser.add_argument("-c", "--dataset-config", type=str, default=".cache/yolov5/coco.yaml")
        parser.add_argument("-s", "--img-size", nargs="+", type=int, default=[640, 640])
        parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value." )
        parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
        # fmt: on
        return parser.parse_args()

    def __init__(self) -> None:
        args = self.get_args()
        self.video: str = args.video
        self.output_dir: str = args.outdir
        self.model_path: str = args.model
        self.dataset_config: str = args.dataset_config
        if len(args.img_size) == 2:
            self.img_size: List[int] = args.img_size
        elif len(args.img_size) == 1:
            self.img_size: List[int] = args.img_size * 2
        else:
            raise ValueError("Invalid img_size")
        self.aspect_ratio_thresh: float = args.aspect_ratio_thresh
        self.min_box_area: float = args.min_box_area

        if not os.path.exists(self.video):
            raise FileNotFoundError(f"Video file {self.video} not found")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found")


def main():
    args = TrackArgs()
    video_path = args.video
    output_dir = args.output_dir

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_strlen = len(str(frame_count))

    # Load detector / 加载检测器
    ovbackend = DetectorInferBackends().OpenVINOBackend
    print("-- Available devices:", ovbackend.query_device())
    detector = ovbackend(device="AUTO")
    detector.load_model(args.model_path, verbose=True)

    with open(args.dataset_config, "r") as f:
        label_map: Dict[int, str] = yaml.load(f, Loader=yaml.FullLoader)[
            "names"
        ]
        label_list = list(label_map.values())
        print(label_list)

    tracker = ByteTracker()
    img_size = args.img_size

    async_preds = None

    def async_completion_callback(infer_request, info):
        global async_preds
        # input: np.ndarray = info
        print(info)
        start_time = info["start_time"]
        infer_time = get_consume_t_ms(start_time)
        async_preds = infer_request.get_output_tensor(0).data[0]
        print(f"Async infer completed in {infer_time:.2f} ms")

        # TODO: postprocess

    detector.enable_async_mode_infer(async_completion_callback)

    frame_id = 0
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id > 50:
            break

        start_time = time.time()
        img = frame
        debug_img = copy.deepcopy(frame)
        copy_time = get_consume_t_ms(start_time)
        start_time = time.time()

        input_t, scale_h, scale_w = Process.preprocess(img, img_size)
        preprocess_time = get_consume_t_ms(start_time)
        start_time = time.time()

        detector.infer_async(
            input_t,
            **dict(
                start_time=start_time,
                frame_id=frame_id,
            ),
        )
        continue

        infer_time = get_consume_t_ms(start_time)
        start_time = time.time()

        preds = Process.postprocess(
            output_t
        )  # [ B, [x1, y1, x2, y2, conf, cls] ]

        online_targets = tracker.update(preds, scale_h, scale_w)
        online_tackbboxes: List[TTrackBbox] = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tcls = t.clsid
            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tackbboxes.append(TTrackBbox(tlwh, tid, t.score, tcls))

        postprocess_time = get_consume_t_ms(start_time)

        display_info = " ".join(
            [
                f"Frame {str(frame_id).rjust(frame_count_strlen)}/{frame_count}",
                f"Copy: {copy_time:.2f}",
                f"Preprocess: {preprocess_time:.2f}",
                f"Infer: {infer_time:.2f}",
                f"Postprocess: {postprocess_time:.2f}",
                "(ms)",
            ]
        )

        print(display_info)
        display_info = " ".join(
            [
                f"Frame {str(frame_id).rjust(frame_count_strlen)}/{frame_count}",
                f"Time: {preprocess_time+infer_time+postprocess_time:.2f}(ms)",
            ]
        )

        online_im = plot_tracking(
            debug_img,
            online_tackbboxes,
            label_list,
            display_info=display_info,
        )

        # Process.mark(debug_img, preds, label_list, scale_h, scale_w)

        cv2.imwrite(f"{output_dir}/out.jpg", online_im)
        # cv2.imshow("frame", online_im)
        # cv2.waitKey(10)
    cv2.destroyAllWindows()


def get_consume_t_ms(start_time: float):
    return (time.time() - start_time) * 1000


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(
    img: np.ndarray,
    tackbboxes: List[TTrackBbox],
    label_list: List[str],
    ids2=None,
    display_info: str = "",
):
    img = np.ascontiguousarray(np.copy(img))
    img_h, img_w = img.shape[:2]

    top_view = np.zeros([img_w, img_w, 3], dtype=np.uint8) + 255
    text_scale = 1.5
    text_thickness = 2
    line_thickness = 2
    radius = max(5, int(img_w / 140.0))

    cv2.putText(
        img,
        display_info,
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 255),
        thickness=2,
    )

    for i, tackbbox in enumerate(tackbboxes):
        x1, y1, w, h = tackbbox.tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(tackbbox.objid)
        id_text = (
            f"{label_list[tackbbox.clsid]}/{int(obj_id)} ({tackbbox.score:.2f})"
        )
        color = get_color(abs(obj_id))
        cv2.rectangle(
            img, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
        )
        cv2.putText(
            img,
            id_text,
            (intbox[0], intbox[1] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 255),
            thickness=text_thickness,
        )

    return img


if __name__ == "__main__":
    main()
