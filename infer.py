import argparse
import os
from typing import Dict, List
import cv2
import tqdm
import yaml

from dlinfer.detector import DetectorInferBackends
from dlinfer.detector import Process


class InferArgs:
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(add_help=False)
        # fmt: off
        parser.add_argument("-m", "--model", type=str, default=".cache/yolov5/yolov5s.onnx")
        parser.add_argument("-c", "--config", type=str, default=".cache/yolov5/coco.yaml")
        parser.add_argument("-s", "--img-size", nargs="+", type=int, default=[640, 640])
        parser.add_argument("-i", "--input", type=str, default="images/bus.jpg")
        # fmt: on
        return parser.parse_args()

    def __init__(self) -> None:
        args = self.get_args()
        self.model: str = args.model
        self.config: str = args.config
        if len(args.img_size) == 2:
            self.img_size: List[int] = args.img_size
        elif len(args.img_size) == 1:
            self.img_size: List[int] = [args.img_size, args.img_size]
        else:
            raise ValueError("Invalid img_size")
        self.input: str = args.input


def main() -> int:
    args = InferArgs()
    backends = DetectorInferBackends()
    # =============== Choose backend to Infer ===============
    # ------ Choose one and comment out the others ------
    ## ------ ONNX ------
    # onnx_backend = backends.ONNXBackend
    # print("-- Available devices:", providers := onnx_backend.SUPPORTED_DEVICES)
    # detector = onnx_backend(device=providers, inputs=["images"], outputs=["output0"])

    ## ------ OpenVINO ------
    ov_backend = backends.OpenVINOBackend
    print("-- Available devices:", ov_backend.query_device())
    detector = ov_backend(device="AUTO")

    ## ------ TensorRT ------
    # detector = backends.TensorRTBackend()
    # =======================================================

    detector.load_model(args.model, verbose=True)

    with open(args.config, "r") as f:
        file_content = yaml.load(f, Loader=yaml.FullLoader)
    label_map: Dict[int, str] = file_content["names"]
    label_list = list(label_map.values())

    img = cv2.imread(args.input)  # H W C
    os.makedirs("tmp", exist_ok=True)

    img_size = args.img_size
    # -- warm up
    input_t, scale_h, scale_w = Process.preprocess(img, img_size)  # B C H W
    output_t = detector.infer(input_t)

    # -- do inference
    print("-- do inference")
    pbar = tqdm.tqdm(range(10))
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
        end_time = cv2.getTickCount()
        total_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
        total_sum_time += total_time
        _cnt += 1
        pbar.set_description(
            f"Time infer/total: {infer_time:.2f}/{total_time:.2f} ms"
        )
    # -- mark
    print(f"-- Average time: {total_sum_time / _cnt:.2f} ms")
    Process.mark(img, preds, label_list, scale_h, scale_w)
    cv2.imwrite(save_path := "tmp/out.jpg", img)
    print(f"-- output saved to '{save_path}'")


if __name__ == "__main__":
    main()
