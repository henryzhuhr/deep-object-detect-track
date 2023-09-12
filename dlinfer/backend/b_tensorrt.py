from typing import List
import numpy as np
from .interface import IBackend
import tensorrt as trt

print(trt.__version__)


# cuda-python for tensorrt >= 8.0

from cuda import cuda, cudart


class TensorRTBackend(IBackend):
    NAME = "TensorRT"
    SUPPORTED_VERISONS = ["8.6"]
    SUPPORTED_DEVICES = ["CPU", "GPU", "MYRIAD", "HDDL", "HETERO"]

    def __init__(self, device="GPU") -> None:
        trt_version: str = trt.__version__
        super().__init__(trt_version)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine: trt.ICudaEngine = None

    def load_model(self, model_path: str, verbose: bool = False) -> None:
        try:
            # Load TRT engine
            with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                assert runtime
                self.engine = runtime.deserialize_cuda_engine(f.read())
            assert self.engine
            self.context = self.engine.create_execution_context()
            assert self.context

            # Setup I/O bindings
            self.inputs = []
            self.outputs = []
            self.allocations = []

            for i in range(self.engine.num_bindings):
                is_input = False
                if self.engine.binding_is_input(i):
                    is_input = True
                name = self.engine.get_binding_name(i)
                dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
                shape = self.context.get_binding_shape(i)

                if is_input and shape[0] < 0:
                    assert self.engine.num_optimization_profiles > 0
                    profile_shape = self.engine.get_profile_shape(0, name)
                    assert len(profile_shape) == 3  # min,opt,max
                    # Set the *max* profile as binding shape
                    self.context.set_binding_shape(i, profile_shape[2])
                    shape = self.context.get_binding_shape(i)

                if is_input:
                    self.batch_size = shape[0]
                size = dtype.itemsize
                for s in shape:
                    size *= s

                allocation = cuda_call(cudart.cudaMalloc(size))
                host_allocation = None if is_input else np.zeros(shape, dtype)
                binding = {
                    "index": i,
                    "name": name,
                    "dtype": dtype,
                    "shape": list(shape),
                    "allocation": allocation,
                    "host_allocation": host_allocation,
                }
                self.allocations.append(allocation)
                if self.engine.binding_is_input(i):
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)

                if verbose:
                    print(
                        " -- [{}] '{}' with dtype {} and shape {}".format(
                            "Input" if is_input else "Output",
                            self.ColorStr.info(binding["name"]),
                            self.ColorStr.info(binding["dtype"]),
                            self.ColorStr.info(binding["shape"]),
                        )
                    )

            assert self.batch_size > 0
            assert len(self.inputs) > 0
            assert len(self.outputs) > 0
            assert len(self.allocations) > 0
        except Exception as e:
            raise RuntimeError("Failed to load model due to: {}".format(e))

    def infer(self, input: np.ndarray) -> np.ndarray:
        # Copy I/O and Execute
        memcpy_host_to_device(self.inputs[0]["allocation"], np.ascontiguousarray(input))
        self.context.execute_v2(self.allocations)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            memcpy_device_to_host(self.outputs[o]["host_allocation"], self.outputs[o]["allocation"])
        outputs: List[np.ndarray] = [o["host_allocation"] for o in self.outputs]
        output = outputs[0]
        return output


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
    )


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
    )


def cuda_call(call):
    def check_cuda_err(err):
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
        if isinstance(err, cudart.cudaError_t):
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("Cuda Runtime Error: {}".format(err))
        else:
            raise RuntimeError("Unknown error type: {}".format(err))

    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res
