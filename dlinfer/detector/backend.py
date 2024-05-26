import os
import warnings


class DetectorInferBackends:
    def __init__(self) -> None:
        try:
            from .b_onnx import ONNXDetectorBackend

            self.ONNXBackend = ONNXDetectorBackend
        except ImportError as e:
            self.ONNXBackend = None
            # fmt: off
            warning_message = "ONNXBackend is not available. Please install onnxruntime-gpu or onnxruntime."
            # fmt: on
            warnings.warn(f"\033[00;33m{warning_message}\033[0m")
            print(f"\033[00;33m{e}\033[0m", os.linesep)

        try:
            from .b_openvino import OpenVINODetectorBackend

            self.OpenVINOBackend = OpenVINODetectorBackend
        except ImportError as e:
            self.OpenVINOBackend = None
            # fmt: off
            warning_message = "OpenVINOBackend is not available. Please install openvino."
            # fmt: on
            warnings.warn(f"\033[00;33m{warning_message}\033[0m")
            print(f"\033[00;33m{e}\033[0m", os.linesep)

        try:
            from .b_tensorrt import TensorRTDetectorBackend

            self.TensorRTBackend = TensorRTDetectorBackend
        except ImportError as e:
            self.TensorRTBackend = None
            # fmt: off
            warning_message = "TensorRTBackend is not available. Please install tensorrt."
            # fmt: on
            warnings.warn(f"\033[00;33m{warning_message}\033[0m")
            print(f"\033[00;33m{e}\033[0m", os.linesep)
