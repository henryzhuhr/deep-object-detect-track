from abc import ABC, abstractmethod
import numpy as np

class IBackend(ABC):
    class ColorStr:
        @staticmethod
        def info(str_info: str):
            # GREEN=$(echo -en '\033[00;32m')
            return f"\033[00;32m{str_info}\033[0m"

        @staticmethod
        def error(str_info):
            # RED=$(echo -en '\033[00;31m')
            return f"\033[00;31m{str_info}\033[0m"

        @staticmethod
        def warning(str_info):
            # YELLOW=$(echo -en '\033[00;33m')
            return f"\033[00;33m{str_info}\033[0m"

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def infer(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError