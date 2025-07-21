from abc import ABC, abstractmethod
from typing import List

import numpy as np
from loguru import logger

DEFAULT_NAME = "IDetector"


class IDetector(ABC):
    NAME = DEFAULT_NAME
    """name of inference backend, must be rewritten in subclass for checking"""

    SUPPORTED_VERISONS: List[str] = []
    """supported versions of inference backend, must be rewritten in subclass for checking version"""

    SUPPORTED_DEVICES: List[str] = []
    """supported devices of inference backend, must be rewritten in subclass for checking device"""

    def __init__(self, version: str) -> None:
        super().__init__()
        if self.NAME == DEFAULT_NAME:
            raise NotImplementedError("NAME must be rewritten in subclass")

        if len(self.SUPPORTED_VERISONS) < 1:
            logger.warning("SUPPORTED_VERISONS must be rewritten in subclass")

        if len(self.SUPPORTED_DEVICES) < 1:
            logger.warning("SUPPORTED_DEVICES must be rewritten in subclass")

        self._check_version(version)

    def _check_version(self, version: str):
        """
        Check if the version of inference backend is supported
        :param version: version of inference backend
        """
        for sv in self.SUPPORTED_VERISONS:
            if version.startswith(sv):
                return
        logger.warning(
            f"{self.NAME} version {version} is not supported, "
            f"please upgrade to support version: {self.SUPPORTED_VERISONS}"
        )

    @abstractmethod
    def load_model(self, model_path: str, verbose: bool = False) -> None:
        """
        Load model from model_path, must be rewritten in subclass
        :param model_path: path to model
        """
        raise NotImplementedError("IDetector.load_model must be rewritten in subclass")

    @abstractmethod
    def infer(self, input: np.ndarray) -> np.ndarray:
        """
        Do inference, must be rewritten in subclass
        :param input: input tensor
        :return: output tensor
        """
        raise NotImplementedError("IDetector.infer must be rewritten in subclass")
