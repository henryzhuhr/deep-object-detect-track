import time
from .interface import IDetectoBackends
import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor
from openvino.runtime import (
    __version__,
    AsyncInferQueue,
    Core,
    CompiledModel,
    InferRequest,
    Layout,
    Type,
    Model,
)


class OpenVINODetectorBackend(IDetectoBackends):
    NAME = "OpenVINO"
    SUPPORTED_VERISONS = ["2023.0.1", "2024.1.0"]
    SUPPORTED_DEVICES = ["CPU", "GPU", "MYRIAD", "HDDL", "HETERO"]

    def __init__(self, device="AUTO") -> None:
        ov_version: str = __version__
        super().__init__(ov_version)
        self.core: Core = Core()
        self.device = device
        self.model: CompiledModel = None
        self.infer_request = None
        self.async_mode = False
        self.curr_request = None
        self.next_request = None
        # TODO： [Multi-device execution](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/multi-device.html)

    def load_model(self, model_path: str, verbose: bool = False) -> None:
        # fmt: off
        model: Model = self.core.read_model(model_path)
        assert len(model.inputs) == 1, "Sample supports only single input topologies"
        assert len(model.outputs) == 1, "Sample supports only single output topologies"
        
        # -- Apply preprocessing
        # https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html
        ppp = PrePostProcessor(model)

        # 1) Set input tensor information:
        # - input() provides information about a single model input
        # - precision of tensor is supposed to be 'u8'
        # - layout of data is 'NHWC'
        # ppp.input().tensor().set_element_type(Type.u8).set_layout(
        #     Layout("NHWC")
        # )  # noqa: N400

        # 2) Here we suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout(Layout("NCHW"))

        # 3) Set output tensor information:
        # - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(Type.f32)

        # 4) Apply preprocessing modifing the original 'model'
        model = ppp.build()

        # -- Loading model to the device
        compiled_model = self.core.compile_model(model, self.device)
        self.model = compiled_model
        self.infer_request = compiled_model.create_infer_request()
        self.input_layer_ir = model.input(0)
        # fmt: on

    def enable_async_mode_infer(self, completion_callback):
        infer_queue = AsyncInferQueue(
            self.model, jobs=2
        )  # API: https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.runtime.AsyncInferQueue.html
        infer_queue.set_callback(completion_callback)
        self.infer_queue = infer_queue
        self.async_mode = True

    def infer(self, input: np.ndarray) -> np.ndarray:
        """
        # 执行模型推理，对 ONNX 模型进行推理的封装
        ## Args:
        - `input`: `np.ndarray`, `(1, C, H, W)`, `np.float32`, range: `[0, 1]`

        ## Returns:
        - `output`: `np.ndarray`, `(201, 18, 4)`, `np.float32`, range: `[0, 1]`
        """
        # `input name`  is compatible with ONNX model `binding__input_names`
        # `output name` is compatible with ONNX model `binding_output_names`
        # outputs: List[np.ndarray] = self.ort_session.run(["output"], {"input": input})
        # output = outputs[0]
        # return output
        input_layer_ir = self.input_layer_ir
        infer_request = self.infer_request
        infer_request.set_tensor(input_layer_ir, ov.Tensor(input))
        infer_request.infer()
        preds = infer_request.get_output_tensor(0).data
        return preds

    def infer_async(self, input: np.ndarray, **kwargs) -> np.ndarray:
        """
        https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/inference-request.html#asynchronous-mode
        """
        infer_queue = self.infer_queue
        input_layer_ir = self.input_layer_ir
        infer_queue.start_async(
            {input_layer_ir.any_name: input},
            (kwargs),
        )

    @staticmethod
    def query_device():
        """Query available devices for OpenVINO backend."""
        return Core().available_devices


# input_layer_ir = model.input(0)
# N, C, H, W = input_layer_ir.shape
# shape = (H, W)
