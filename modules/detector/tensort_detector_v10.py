from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorrt as trt
from cuda import cuda, cudart
from loguru import logger

from modules.detector.tensorrt_utils import (
    TensorRTVersionInfo,
    support_trt_version_list,
)

from .tensort_detector import ITensorRTDetector

DEFAULT_ITensorRTDetector_NAME = "ITensorRTDetector"


class common:
    @staticmethod
    def check_cuda_err(err):
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
        if isinstance(err, cudart.cudaError_t):
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("Cuda Runtime Error: {}".format(err))
        else:
            raise RuntimeError("Unknown error type: {}".format(err))

    @staticmethod
    def cuda_call(call):
        err, res = call[0], call[1:]
        common.check_cuda_err(err)
        if len(res) == 1:
            res = res[0]
        return res

    @staticmethod
    # Wrapper for cudaMemcpy which infers copy size and does error checking
    def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
        nbytes = host_arr.size * host_arr.itemsize
        # fmt: off
        common.cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        # fmt: on

    @staticmethod
    # Wrapper for cudaMemcpy which infers copy size and does error checking
    def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
        nbytes = host_arr.size * host_arr.itemsize
        # fmt: off
        common.cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        # fmt: on


class TensorRTDetectorV10(ITensorRTDetector):
    """
    Base class for TensorRT-based detectors.
    This class extends the IDetector interface and provides a structure for TensorRT detectors.
    It must be implemented by any TensorRT detector class.
    It includes methods for loading models and performing inference, which must be implemented in subclasses.
    It also includes attributes for the name, supported versions, and devices of the detector.
    It is designed to be used as a base class for specific TensorRT detector implementations.
    """

    name = "TensorRTDetector_v10"
    supported_version = TensorRTVersionInfo.get_support_version(
        support_trt_version_list
    )
    supported_devices = []  # TensorRT must rely on CUDA

    # Setup I/O bindings
    inputs: List[dict]
    outputs: List[dict]
    allocations: List[int]

    def __init__(self) -> None:
        super().__init__()

        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine: Optional[trt.ICudaEngine] = None

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []

    def load_model(self, model_path: str, verbose: bool = False) -> None:
        p = Path(model_path).expanduser().resolve()
        # Load TRT engine
        with open(p, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime  # tensorrt.tensorrt.Runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))

            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": shape,  # list(shape),
                "allocation": allocation,
            }

            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

            if verbose:
                logger.info(
                    "[{}] '{}' with dtype {} and shape {}".format(
                        "Input" if is_input else "Output",
                        binding["name"],
                        binding["dtype"],
                        binding["shape"],
                    )
                )

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, input: np.ndarray) -> np.ndarray:
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        - param `input`: A numpy array holding the image batch.
        """
        # Prepare the output data
        output = np.zeros(*self.output_spec())

        # Process I/O and execute the network

        #   1. Copy input to device
        common.memcpy_host_to_device(
            self.inputs[0]["allocation"], np.ascontiguousarray(input)
        )

        #   2. Execute
        # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html#tensorrt.IExecutionContext.execute_v2
        self.context.execute_v2(self.allocations)

        #   3. Copy output to host
        common.memcpy_device_to_host(output, self.outputs[0]["allocation"])

        return output
