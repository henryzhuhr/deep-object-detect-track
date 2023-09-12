from typing import List
from .interface import IBackend
import numpy as np
import onnxruntime as ort


available_providers = ort.get_available_providers()
all_providers = ort.get_all_providers()


class ONNXBackend(IBackend):
    NAME = "ONNX"
    SUPPORTED_VERISONS = ["1.8.0"]
    SUPPORTED_DEVICES = available_providers

    def __init__(
        self,
        device: str = "CPUExecutionProvider",
        inputs: List[str] = ["input"],  # TODO in case of multiple inputs
        outputs: List[str] = ["output"],  # TODO in case of multiple outputs
    ) -> None:
        if device.lower() == "cpu":
            device = "CPUExecutionProvider"
        assert device in self.SUPPORTED_DEVICES, (
            f"specify device {device} is not supported, "
            f"please choose one of supported device: {self.SUPPORTED_DEVICES}"
        )
        self.providers = [device]
        self.ort_session: ort.InferenceSession = None

    def load_model(self, model_path: str, verbose: bool = False) -> None:
        try:
            self.ort_session = ort.InferenceSession(model_path, providers=self.providers)
            binding__input_names = [binding.name for binding in self.ort_session.get_inputs()]
            binding__input_shapes = [binding.shape for binding in self.ort_session.get_inputs()]
            binding__input_types = [binding.type for binding in self.ort_session.get_inputs()]

            binding_output_names = [binding.name for binding in self.ort_session.get_outputs()]
            binding_output_shapes = [binding.shape for binding in self.ort_session.get_outputs()]
            binding_output_types = [binding.type for binding in self.ort_session.get_outputs()]
            if verbose:
                print(self.ColorStr.info("Parsing ONNX info:"))
                print(self.ColorStr.info("  - providers:"), self.ort_session.get_providers())
                print(self.ColorStr.info("  --- inputs:"), binding__input_names)
                print(self.ColorStr.info("       -- names:"), binding__input_names)
                print(self.ColorStr.info("       - shapes:"), binding__input_shapes)
                print(self.ColorStr.info("       -- types:"), binding__input_types)
                print(self.ColorStr.info("  --- outputs:"), binding_output_names)
                print(self.ColorStr.info("       -- names:"), binding_output_shapes)
                print(self.ColorStr.info("       - shapes:"), binding_output_shapes)
                print(self.ColorStr.info("       -- types:"), binding_output_types)

            if "input" not in [binding.name for binding in self.ort_session.get_inputs()]:
                raise ValueError(
                    "'input'  not found in ONNX model, expected one of",
                    binding__input_names,
                )
            if "output" not in [binding.name for binding in self.ort_session.get_outputs()]:
                raise ValueError(
                    "'output' not found in ONNX model, expected one of",
                    binding_output_names,
                )
        except Exception as e:
            raise RuntimeError("Failed to load model due to: {}".format(e))

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
        outputs: List[np.ndarray] = self.ort_session.run(["output"], {"input": input})
        output = outputs[0]
        return output
