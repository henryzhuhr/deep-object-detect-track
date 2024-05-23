from typing import List
import numpy as np
from .interface import IDetectoBackends

import tensorrt as trt

support_trt_version = {
    "8.6": "8.6",
}

if trt.__version__ >= support_trt_version["8.6"]:
    from cuda import cuda, cudart  # cuda-python

    class CUDA_Utils:
        @staticmethod
        def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
            # Wrapper for cudaMemcpy which infers copy size and does error checking
            nbytes = host_arr.size * host_arr.itemsize
            cudart.cudaMemcpy(
                device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )

        @staticmethod
        def memcpy_device_to_host(outputs_bindings: List[dict]):
            # Wrapper for cudaMemcpy which infers copy size and does error checking
            for o in range(len(outputs_bindings)):
                host_arr: np.ndarray = outputs_bindings[o]["host_allocation"]
                device_ptr: int = outputs_bindings[o]["allocation"]
                nbytes = host_arr.size * host_arr.itemsize
                cudart.cudaMemcpy(
                    host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
                )
            outputs: List[np.ndarray] = [o["host_allocation"] for o in outputs_bindings]
            return outputs

        @staticmethod
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

        @staticmethod
        def get_allocation(size):
            return CUDA_Utils.cuda_call(cudart.cudaMalloc(size))

elif trt.__version__ == support_trt_version["8.2"]:
    import pycuda.driver as cuda

    # Use autoprimaryctx if available (pycuda >= 2021.1) to
    # prevent issues with other modules that rely on the primary
    # device context.
    try:
        import pycuda.autoprimaryctx
    except ModuleNotFoundError:
        import pycuda.autoinit
else:
    raise RuntimeError(
        f"Unsupported TensorRT version: {trt.__version__}"
        f"supported versions: {support_trt_version}"
    )


class TensorRTDetectorBackend(IDetectoBackends):
    NAME = "TensorRT"
    SUPPORTED_VERISONS = support_trt_version.values()
    SUPPORTED_DEVICES = ["GPU"]

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
            self.inputs: List[dict] = []
            self.outputs: List[dict] = []
            self.allocations: List[int] = []

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

                allocation = CUDA_Utils.get_allocation(size)
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

        # 1. Copy input to device
        CUDA_Utils.memcpy_host_to_device(self.inputs[0]["allocation"], np.ascontiguousarray(input))

        # 2. Execute
        # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html#tensorrt.IExecutionContext.execute_v2
        self.context.execute_v2(self.allocations)

        # 3. Copy output to host
        outputs = CUDA_Utils.memcpy_device_to_host(self.outputs)
        output = outputs[0]
        return output
