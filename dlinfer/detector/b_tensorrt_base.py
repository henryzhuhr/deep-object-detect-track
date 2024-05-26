from typing import List

import numpy as np
import tensorrt as trt


from .interface import IDetectoBackends
from .trt_utils import TensorRTVersionInfo, support_trt_version_list


class BaseTensorRTDetectorBackend(IDetectoBackends):
    NAME = "TensorRT"
    SUPPORTED_VERISONS = TensorRTVersionInfo.get_support_version(support_trt_version_list)
    SUPPORTED_DEVICES = []  # TensorRT must rely on CUDA

    def __init__(self) -> None:
        trt_version: str = trt.__version__
        super().__init__(trt_version)

        # Setup I/O bindings
        self.inputs: List[dict] = []
        self.outputs: List[dict] = []
        self.allocations: List[int] = []

    def load_model(self, model_path: str, verbose: bool = False) -> None:
        raise NotImplementedError

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        raise NotImplementedError
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        raise NotImplementedError
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, input: np.ndarray) -> np.ndarray:
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        - param `input`: A numpy array holding the image batch.
        """
        raise NotImplementedError
