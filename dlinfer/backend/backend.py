from dlinfer.backend.interface import IBackend


class InferBackends:
    def __init__(self) -> None:
        try:
            from .b_onnx import ONNXBackend
        except ImportError:
            ONNXBackend = None
        self.ONNXBackend = ONNXBackend

        try:
            from .b_openvino import OpenVINOBackend
        except ImportError:
            OpenVINOBackend = None
        self.OpenVINOBackend= OpenVINOBackend

        try:
            from .b_tensorrt import TensorRTBackend
        except ImportError:
            TensorRTBackend = None
        self.TensorRTBackend = TensorRTBackend
