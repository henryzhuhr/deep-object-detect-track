---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# 项目初始化及环境准备

## 获取代码

::: code-group

```bash [SSH(Recommend)]
# 需要配置 github 上的 SSH key
git clone git@github.com:HenryZhuHR/deep-object-detect-track.git
```

```bash [HTTP]
git clone https://github.com/HenryZhuHR/deep-object-detect-track.git
```

:::

进入项目目录

```bash
cd deep-object-detect-track
```

## 系统要求

### 操作系统

项目在 Linux(Ubuntu) 和 macOS 系统并经过测试 ，经过测试的系统：

- ✅ **Ubuntu 22.04.4 LTS jammy** (CPU: 13th Gen Intel(R) Core(TM) i9-13900K & GPU: NVIDIA GeForce RTX 4090)
- ✅ **macOS Tahoe 26** (CPU & GPU: Apple Silicon M1 Pro)

::: warning
项目不支持 Windows 系统 ❌ ，如果需要在 Windows 系统上运行，可以使用 WSL2 或者根据提供的脚本手动执行；虽然已经测试通过，但是不保证所有功能都能正常运行
:::

### GPU

如果需要使用 GPU 训练模型，需要安装 CUDA Toolkit，可以参考 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) 下载对应版本的 CUDA Toolkit，具体下载的版本需要参考 [*INSTALL PYTORCH*](https://pytorch.org/get-started/locally/)

例如 Pytorch 2.3.0 支持 CUDA 11.8/12.1，因此安装 CUDA 11.8/12.1 即可，而不需要过高的 CUDA 版本，安装后需要配置环境变量

```bash
# ~/.bashrc
export CUDA_VERSION=12.1
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

> 事实上，Pytorch 1.8 开始就会在安装的时候自动安装对应的 CUDA Toolkit，因此不需要手动安装 CUDA Toolkit，因此可以跳过这一步

MacOS 系统不支持 CUDA Toolkit，可以使用 CPU 训练模型 (Yolov5 项目暂不支持 MPS 训练)，但是推理过程可以使用 Metal ，参考 [*Introducing Accelerated PyTorch Training on Mac*](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/#getting-started) 和 [*MPS backend*](https://pytorch.org/docs/stable/notes/mps.html#mps-backend)

## 安装环境

这里安装的环境指的是需要训练的环境，如果不需要训练而是直接部署，请转至 「[模型部署](./deploy)」 文档

使用 [uv](https://docs.astral.sh/uv/) 作为项目管理，可以使用命令

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

为了确保 Python 库与系统库一致性，项目中已经强制使用系统的 Python，参考 [Using existing Python versions](https://docs.astral.sh/uv/guides/install-python/#using-existing-python-versions)

```bash
export UV_PYTHON_PREFERENCE=only-system 
uv python install
```

安装依赖

```bash
uv sync
```

如果出现超时的错误 `Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT (current value: 30s)`，可以尝试执行：

```bash
export UV_HTTP_TIMEOUT=3600
uv sync
```

## TensorRT 类型提示

参考 [*Python API autocomplete · Issue #1714 · NVIDIA/TensorRT*](https://github.com/NVIDIA/TensorRT/issues/1714)

```bash
uv sync --extra dev
```

```bash
pybind11-stubgen tensorrt
pybind11-stubgen --ignore-all-errors tensorrt
# pybind11-stubgen --ignore-all-errors tensorrt_bindings # 如果 `from tensorrt_bindings import *` 需要使用这个命令
```

```bash
pybind11-stubgen cuda
```

在 `.vscode/settings.json` 中添加以下配置：

```json
{
  "python.analysis.extraPaths": ["./stubs"],
  "python.analysis.typeCheckingMode": "basic"
}
```
