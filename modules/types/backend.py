from abc import ABC, abstractmethod
from typing import List

import numpy as np
from loguru import logger


class BaseDetectBackend(ABC):
    name: str
    """name of inference backend, must be rewritten in subclass for checking"""

    supported_versions: List[str]
    """supported versions of inference backend, must be rewritten in subclass for checking version"""

    supported_devices: List[str]
    """supported devices of inference backend, must be rewritten in subclass for checking device"""

    def __init__(self, version: str) -> None:
        super().__init__()
        self._check_version(version)

    def _check_version(self, version: str):
        """
        Check if the version of inference backend is supported
        :param version: version of inference backend
        """
        for v in self.supported_versions:
            if version.startswith(v):
                return
        logger.warning(
            f"{self.name} version {version} is not supported, "
            f"please upgrade to support version: {self.supported_versions}"
        )

    @abstractmethod
    def load_model(self, model_path: str, verbose: bool = False):
        """
        Load model from model_path, must be rewritten in subclass
        :param model_path: path to model
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, input: np.ndarray) -> np.ndarray:
        """
        Do inference, must be rewritten in subclass
        :param input: input tensor
        :return: output tensor
        """
        raise NotImplementedError
