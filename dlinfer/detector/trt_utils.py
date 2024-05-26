from typing import List

from ..utils.version import VersionUtils


class TensorRTVersionInfo:
    def __init__(
        self,
        version: str,
        name: str,
        vcuda: List[str] = [],  # supported cuda version list
        vcudnn: List[str] = [],  # supported cudnn version list
        url: str = "",  # package download url
        whl: str = "",  # whl file path
    ) -> None:
        self.version = version
        self.name = name
        self.vcuda = vcuda
        self.vcudnn = vcuda
        self.url = url
        self.whl = whl

    @staticmethod
    def get_support_version(trtv_list: List["TensorRTVersionInfo"]):
        return [trtv.version for trtv in trtv_list]


support_trt_version_list = [
    TensorRTVersionInfo(
        version="10.0",
        name="TensorRT 10.0 GA for Linux x86_64 and CUDA 12.0 to 12.4 TAR Package",
        vcuda=[f"12.{i}" for i in range(0, 5)],
        url="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz",
    ),
    TensorRTVersionInfo(
        version="8.6",
        name="TensorRT 8.6 GA for Linux x86_64 and CUDA 12.0 and 12.1 TAR Package",
        vcuda=[f"12.{i}" for i in range(0, 2)],
        url="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz",
        whl=f"TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp{VersionUtils.get_python_vesion()}-none-linux_x86_64.whl",
    ),
]
