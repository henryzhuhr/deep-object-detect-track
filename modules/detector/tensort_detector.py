from typing import List

import numpy as np
import tensorrt as trt

from modules.detector.interface import IDetector
from modules.detector.tensorrt_utils import (
    TensorRTVersionInfo,
    support_trt_version_list,
)

DEFAULT_ITensorRTDetector_NAME = "ITensorRTDetector"


class ITensorRTDetector(IDetector):
    """
    Base class for TensorRT-based detectors.
    This class extends the IDetector interface and provides a structure for TensorRT detectors.
    It must be implemented by any TensorRT detector class.
    It includes methods for loading models and performing inference, which must be implemented in subclasses.
    It also includes attributes for the name, supported versions, and devices of the detector.
    It is designed to be used as a base class for specific TensorRT detector implementations.
    """

    NAME = DEFAULT_ITensorRTDetector_NAME
    SUPPORTED_VERISONS = TensorRTVersionInfo.get_support_version(
        support_trt_version_list
    )
    SUPPORTED_DEVICES = []  # TensorRT must rely on CUDA

    # Setup I/O bindings
    inputs: List[dict]
    outputs: List[dict]
    allocations: List[int]

    def __init__(self) -> None:
        trt_version: str = trt.__version__
        super().__init__(trt_version)

    def load_model(self, model_path: str, verbose: bool = False) -> None:
        raise NotImplementedError

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        raise NotImplementedError

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        raise NotImplementedError

    def infer(self, input: np.ndarray) -> np.ndarray:
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        - param `input`: A numpy array holding the image batch.
        """
        raise NotImplementedError
