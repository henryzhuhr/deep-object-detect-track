
class DetectorInferBackends:
    def __init__(self) -> None:
        try:
            from .b_onnx import ONNXDetectorBackend
            self.ONNXBackend = ONNXDetectorBackend
        except ImportError:
            self.ONNXBackend = None
            # fmt: off
            print("\033[00;33mWarning: ONNXBackend is not available. Please install onnxruntime-gpu or onnxruntime.\033[0m")
            # fmt: on

        try:
            from .b_openvino import OpenVINODetectorBackend
            self.OpenVINOBackend = OpenVINODetectorBackend
        except ImportError:
            self.OpenVINOBackend = None
            # fmt: off
            print("\033[00;33mWarning: OpenVINOBackend is not available. Please install openvino.\033[0m")
            # fmt: on


        try:
            from .b_tensorrt import TensorRTDetectorBackend
            self.TensorRTBackend = TensorRTDetectorBackend
        except ImportError:
            self.TensorRTBackend = None
            # fmt: off
            print("\033[00;33mWarning: TensorRTBackend is not available. Please install tensorrt.\033[0m")
            # fmt: on

