from typing import List
from .interface import IBackend
import numpy as np
import onnxruntime as ort


class ONNXBackend(IBackend):
    def __init__(
        self,
        providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    ) -> None:
        self.providers = providers
        self.ort_session: ort.InferenceSession = None

    def load_model(self, model_path: str) -> None:
        try:
            self.ort_session = ort.InferenceSession(
                model_path, providers=self.providers
            )
            binding__input_names = [
                binding.name for binding in self.ort_session.get_inputs()
            ]
            binding__input_shapes = [
                binding.shape for binding in self.ort_session.get_inputs()
            ]
            binding__input_types = [
                binding.type for binding in self.ort_session.get_inputs()
            ]

            binding_output_names = [
                binding.name for binding in self.ort_session.get_outputs()
            ]
            binding_output_shapes = [
                binding.shape for binding in self.ort_session.get_outputs()
            ]
            binding_output_types = [
                binding.type for binding in self.ort_session.get_outputs()
            ]

            print(self.ColorStr.info("Parsing ONNX info:"))
            print(
                self.ColorStr.info("  - providers:"), self.ort_session.get_providers()
            )
            print(self.ColorStr.info("  --- inputs:"), binding__input_names)
            print(self.ColorStr.info("       -- names:"), binding__input_names)
            print(self.ColorStr.info("       - shapes:"), binding__input_shapes)
            print(self.ColorStr.info("       -- types:"), binding__input_types)
            print(self.ColorStr.info("  --- outputs:"), binding_output_names)
            print(self.ColorStr.info("       -- names:"), binding_output_shapes)
            print(self.ColorStr.info("       - shapes:"), binding_output_shapes)
            print(self.ColorStr.info("       -- types:"), binding_output_types)

            if "input" not in [
                binding.name for binding in self.ort_session.get_inputs()
            ]:
                raise ValueError(
                    "'input'  not found in ONNX model, expected one of",
                    binding__input_names,
                )
            if "output" not in [
                binding.name for binding in self.ort_session.get_outputs()
            ]:
                raise ValueError(
                    "'output' not found in ONNX model, expected one of",
                    binding_output_names,
                )
        except Exception as e:
            raise RuntimeError("Failed to load model: {}".format(e))

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
