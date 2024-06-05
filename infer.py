import argparse
import os
from typing import Dict
import cv2
import tqdm
import yaml

from dlinfer.detector import DetectorInferBackends
from dlinfer.detector import Process


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument("--model", type=str, default=".cache/yolov5/yolov5s.onnx")
    args.add_argument("-i", "--input", type=str, default="images/bus.jpg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    backends = DetectorInferBackends()
    # =============== Choose backend to Infer ===============
    # ------ Choose one and comment out the others ------
    ## ------ ONNX ------
    onnx_backend = backends.ONNXBackend
    print("-- Available devices:", providers := onnx_backend.SUPPORTED_DEVICES)
    detector = onnx_backend(device=providers, inputs=["images"], outputs=["output0"])

    ## ------ OpenVINO ------
    # ov_backend = backends.OpenVINOBackend
    # print("-- Available devices:", ov_backend.query_device())
    # detector = ov_backend(device="GPU.0")

    ## ------ TensorRT ------
    # detector = backends.TensorRTBackend()
    # =======================================================

    detector.load_model(args.model, verbose=True)

    with open(".cache/yolov5/coco.yaml", "r") as f:
        label_map: Dict[int, str] = yaml.load(f, Loader=yaml.FullLoader)["names"]
        label_list = list(label_map.values())
        # print(label_list)

    img = cv2.imread(args.input)  # H W C
    os.makedirs("tmp", exist_ok=True)

    # -- warm up
    input_t, scale_h, scale_w = Process.preprocess(img)  # B C H W
    output_t = detector.infer(input_t)

    # -- do inference
    print("-- do inference")
    pbar = tqdm.tqdm(range(100))
    total_sum_time = 0
    _cnt = 0
    for i in pbar:
        start_time = cv2.getTickCount()
        # -- preprocess
        input_t, scale_h, scale_w = Process.preprocess(img)  # B C H W
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
        pbar.set_description(f"Time infer/total: {infer_time:.2f}/{total_time:.2f} ms")
    # -- mark
    print(f"-- Average time: {total_sum_time / _cnt:.2f} ms")
    Process.mark(img, preds, label_list, scale_h, scale_w)
    cv2.imwrite(save_path := "tmp/out.jpg", img)
    print(f"-- output saved to '{save_path}'")


if __name__ == "__main__":
    main()
