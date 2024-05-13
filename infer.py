import argparse
import os
from typing import Dict
import cv2
import numpy as np
import yaml

from dlinfer import InferBackends, Process


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument(
        "-m",
        "--model",
        type=str,
        default=".cache/yolov5/yolov5s.onnx",
        help="Required. Path to an .xml or .onnx file with a trained model.",
    )
    args.add_argument("-i", "--input", type=str, default="images/bus.jpg")
    args.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Optional. Specify the target device to infer on; CPU, GPU, GNA or HETERO: "
        "is acceptable. The sample will look for a suitable plugin for device specified. "
        "Default value is CPU.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        backends = InferBackends()
        # backend = backends.ONNXBackend(
        #     device=args.device, inputs=["images"], outputs=["output0"]
        # )
        backend = backends.OpenVINOBackend(device="AUTO")
        print(backend.query_device())
    except RuntimeError as e:
        print(e)
        return 1
    backend.load_model(args.model, verbose=True)

    with open("resource/yolov5s.yaml", "r") as f:
        label_map: Dict[int, str] = yaml.load(f, Loader=yaml.FullLoader)[
            "names"
        ]
        label_list = list(label_map.values())
        # print(label_list)

    img = cv2.imread(args.input)  # H W C
    os.makedirs("tmp", exist_ok=True)

    
    # -- warm up
    input_t, scale_h, scale_w = Process.preprocess(img)  # B C H W
    output_t = backend.infer(input_t)

    # -- do inference
    print("-- do inference")
    start_time = cv2.getTickCount()
    input_t, scale_h, scale_w = Process.preprocess(img)  # B C H W
    output_t = backend.infer(input_t)

    end_time = cv2.getTickCount()
    print(
        "Inference time: ",
        (end_time - start_time) / cv2.getTickFrequency() * 1000,
        "ms",
    )

    preds = Process.postprocess(output_t)

    # -- mark
    Process.mark(img, preds, label_list, scale_h, scale_w)
    cv2.imwrite("tmp/out.jpg", img)


if __name__ == "__main__":
    main()
