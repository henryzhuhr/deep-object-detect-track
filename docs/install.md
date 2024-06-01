---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# 安装环境


以 yolov5 为基础，实现目标检测和跟踪算法的训练和部署，部署 TensorRT 和 OpenVINO 两种方案。

## 环境配置

### 获取代码

::: code-group
```shell [HTTP]
git clone --recursive https://github.com/HenryZhuHR/deep-object-detect-track.git
```
```shell [SSH]
git clone --recursive git@github.com:HenryZhuHR/deep-object-detect-track.git
```
::: 

进入项目目录

```shell
cd deep-object-detect-track
```

> 后续的脚本基于 deep-object-detect-track 目录下执行

如果未能获取子模块，可以手动获取
```shell
# in deep-object-detect-track directory
git submodule init
git submodule update
```

### 创建环境并安装依赖

确保安装了 conda ，如果没有安装，请从 [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) 下载，或者快速安装

```shell
# linux x64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```


#### 方法一：使用提供的脚本


查看 `scripts/base.sh` ，根据需要修改配置，执行命令自动创建并且激活虚拟环境

::: code-group

```shell [使用 venv 创建虚拟环境]
bash scripts/python-activate.venv.sh
```

```shell [使用 conda 创建虚拟环境]
bash scripts/python-activate.conda.sh
```

:::

- **可以重复执行该脚本获取激活环境的提示信息或者安装依赖**
- 修改 `export PIP_QUIET=true` 为 `false` 可以查看安装过程
- 该脚本会复制 `yolov5/requirements.txt` 到 `.cache/yolov5/requirements.txt`，可以自行修改 `.cache/yolov5/requirements.txt` 文件安装相关依赖，例如取消 `onnx` 的注释以支持 ONNX 格式的模型导出

#### 方法二：手动安装

创建虚拟环境
```shell
export ENV_NAME=deep-object-detect-track
# 在项目内安装环境(推荐)
conda create -p .env/$ENV_NAME python=3.10 -y
conda activate ./.env/$ENV_NAME
# 全局安装环境
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME
```
> Python 版本选择 3.10 是因为 Ubuntu 22.04 默认安装的 Python 版本是 3.10

1. 安装 PyTorch

参考官网 [INSTALL PYTORCH](https://pytorch.org/get-started/locally/) 选择配置安装 PyTorch

```shell
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121
```
> 链接最后的 `cu121` 是需要根据系统的 CUDA 版本进行选择

接下来安装其他依赖

```shell
pip install -r projects/yolov5/requirements.txt
pip install -r requirements/requirements.train.txt
```

如果涉及部署流程，需要自行修改 `requirements.txt` 文件，将下列依赖取消注释掉，然后重新执行上述命令

```txt
# Export ----------------------------------------------------------------------
# coremltools>=6.0  # CoreML export
# onnx>=1.10.0  # ONNX export
# onnx-simplifier>=0.4.1  # ONNX simplifier
# nvidia-pyindex  # TensorRT export
# nvidia-tensorrt  # TensorRT export
# scikit-learn<=1.1.2  # CoreML quantization
# tensorflow>=2.4.0,<=2.13.1  # TF exports (-cpu, -aarch64, -macos)
# tensorflowjs>=3.9.0  # TF.js export
# openvino-dev>=2023.0  # OpenVINO export
```
- `onnx`: ONNX 格式的模型导出支持任意设备，需要取消注释，并且其他导出依赖于 ONNX 模型
- `coremltools`: 必须依赖于 MacOS 系统
- `nvidia-*`: 确保硬件支持 NVIDIA GPU
