---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# 安装环境


## 获取代码

::: code-group

```shell [SSH(Recommend)]
# 需要配置 github 上的 SSH key
git clone --recursive git@github.com:HenryZhuHR/deep-object-detect-track.git
```

```shell [HTTP]
git clone --recursive https://github.com/HenryZhuHR/deep-object-detect-track.git
```

:::

进入项目目录

```shell
cd deep-object-detect-track
```

> 后续的脚本基于 `deep-object-detect-track` 目录下执行

如果未能获取子模块，可以手动获取，如果 `git submodule` 无法获取，可以使用 `git clone` 获取

::: code-group

```shell [git submodule]
# in deep-object-detect-track directory
git submodule init
git submodule update
```

```shell [git clone]
git clone https://github.com/ultralytics/yolov5.git projects/yolov5
```

:::


## 系统要求

### 操作系统


项目在 Linux(Ubuntu) 和 MacOS 系统并经过测试 ，经过测试的系统：
- ✅ Ubuntu 22.04 jammy (CPU & GPU)
- ✅ MacOS (CPU)

> 项目不支持 Windows 系统 ❌ ，如果需要在 Windows 系统上运行，可以使用 WSL2 或者根据提供的脚本手动执行；虽然已经测试通过，但是不保证所有功能都能正常运行，因此不接受 Windows 系统的问题反馈

### GPU

如果需要使用 GPU 训练模型，需要安装 CUDA Toolkit，可以参考 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) 下载对应版本的 CUDA Toolkit，具体下载的版本需要参考 [*INSTALL PYTORCH*](https://pytorch.org/get-started/locally/)

例如 Pytorch 2.3.0 支持 CUDA 11.8/12.1，因此安装 CUDA 11.8/12.1 即可，而不需要过高的 CUDA 版本，安装后需要配置环境变量

```shell
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

提供两种方式安装， venv 或 conda

- **venv** : 如果没有安装，请安装

::: code-group

```shell [Linux]
sudo apt install -y python3-venv python3-pip
```

```shell [MacOS]
# Mac 貌似自带了 python3-venv
# brew install python3-venv python3-pip
```

:::

- **conda** : 如果没有安装，请从 [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) 下载，或者快速安装

::: code-group

```shell [linux x64]
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

```shell [MacOS arm64]
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
zsh Miniconda3-latest-MacOSX-arm64.sh
```

::: 

### 方法一：手动安装

创建虚拟环境

::: code-group

```shell [在项目内安装环境(推荐)]
conda create -p .env/deep-object-detect-track python=3.10 -y
conda activate ./.env/deep-object-detect-track
```
    
```shell [全局安装环境]
conda create -n deep-object-detect-track python=3.10 -y
conda activate deep-object-detect-track
```

:::

> Python 版本选择 3.10 是因为 Ubuntu 22.04 默认安装的 Python 版本是 3.10

1. 安装 PyTorch

参考官网 [*INSTALL PYTORCH*](https://pytorch.org/get-started/locally/) 选择配置安装 PyTorch

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
> 链接最后的 `cu121` 是需要根据系统的 CUDA 版本进行选择

接下来安装其他依赖

```shell
pip install -r requirements/requirements.yolov5.txt
```




### 方法二：使用提供的脚本

提供的安装脚本依赖于基本环境变量 `scripts/variables.sh` ，可以复制一份到项目目录下进行自定义修改（推荐），如果不需要修改，可以直接执行

```shell
cp scripts/variables.sh scripts/variables.custom.sh
```
- `CACHE_DIR`: 用于存放一些缓存文件，例如 `yolov5/requirements.txt`，默认为项目目录下的 `.cache`
- 安装过程会自动检测 `CUDA_VERSION` 以安装对应的 PyTorch 版本，否则默认安装 CPU 版本的 PyTorch；如果电脑有 NVIDIA GPU 但是不想安装 CUDA Toolkit 到全局系统（需要 sudo）可以取消注释 `export CUDA_VERSION=12.1` 以安装对应的 PyTorch 版本

运行会自动检测是否存在用户自定义的环境变量 `scripts/variables.custom.sh` ，如果存在则使用自定义的环境变量，否则使用默认的环境变量 `scripts/variables.sh` 

执行命令自动创建并且激活虚拟环境，默认使用 `venv`，**可以重复执行该脚本获取激活环境的提示信息或者安装依赖**

::: code-group

```shell [使用 venv 创建虚拟环境]
bash scripts/create-python-env.sh -i # -i 自动安装依赖
#zsh scripts/create-python-env.sh -i # zsh
```

```shell [使用 conda 创建虚拟环境]
bash scripts/create-python-env.sh -e conda -i # -i 自动安装依赖
#zsh scripts/create-python-env.sh -e conda -i # zsh
```

:::

- 该脚本会复制 `yolov5/requirements.txt` 到 `.cache/yolov5/requirements.txt`，可以自行修改 `.cache/yolov5/requirements.txt` 文件安装相关依赖，例如取消 `onnx` 的注释以支持 ONNX 格式的模型导出；可以修改后再次执行脚本以重新安装依赖


