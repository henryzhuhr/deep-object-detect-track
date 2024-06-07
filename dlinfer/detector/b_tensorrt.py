from os import error
from typing import List

import numpy as np
import tensorrt as trt


trt_version: str = trt.__version__
tv_list: List[str] = trt_version.split(".")
tv_major: str = tv_list[0]
tv_minor: str = tv_list[1]
tv_patch: str = tv_list[2]


if f"{tv_major}.{tv_minor}" in ["10.0"]:
    from .b_tensorrt_10_0 import TensorRTDetectorBackend_10_0

    TensorRTDetectorBackend = TensorRTDetectorBackend_10_0
elif f"{tv_major}.{tv_minor}" in ["8.4", "8.6"]:
    from .b_tensorrt_8_6 import TensorRTDetectorBackend_8_6

    TensorRTDetectorBackend = TensorRTDetectorBackend_8_6
elif f"{tv_major}.{tv_minor}" in ["8.2"]:
    raise NotImplementedError(
        "TensorRT 8.2 is to be implemented in the future only for Jetson Nano device."
    )
    # from .b_tensorrt_8_2 import TensorRTDetectorBackend_8_2
    # TensorRTDetectorBackend = TensorRTDetectorBackend_8_2
else:
    raise NotImplementedError(f"Unsupported TRT version: {trt_version}")
