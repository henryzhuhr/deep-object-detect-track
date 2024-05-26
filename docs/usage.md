---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# 项目使用文档


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

### 创建虚拟环境

确保安装了 conda ，如果没有安装，请从 [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) 下载，或者快速安装
  
```shell
# linux x64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

创建虚拟环境
```shell
# 在项目内安装环境(推荐)
conda create -p .env/dodt python=3.10 -y
conda activate ./.env/dodt
# 全局安装环境
conda create -n dodt python=3.10 -y
conda activate dodt
```
> Python 版本选择 3.10 是因为 Ubuntu 22.04 默认安装的 Python 版本是 3.10

### 安装依赖

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
pip install -r requirements.txt
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

### 下载预训练模型

在 [YOLOv5 Releases](https://github.com/ultralytics/yolov5/releases) 页面下载预训练模型，放置在 `weights` 目录下


## 训练模型

## 导出模型

已经编写了一个脚本，可以直接导出

```shell
# 查看脚本，修改参数
bash scripts/export-yolov5.sh
```

也可以自行导出，进入 yolov5 项目目录

```shell
cd projects/yolov5
```
设定参数，执行
```shell
python3 export.py \
    --weights ../weights/yolov5s.pt \
    --data data/coco128.yaml \
    --include onnx openvino 
```
- `--weights` 模型路径:
- `--include` 训练时数据集参数:
- `--data` 导出类型: 可以导出多个模型，用空格分隔

## 部署模型

部署模型请单独创建虚拟环境，并根据部署中提到的依赖进行**最小化安装**，而避免引入过多不必要的包 （例如 pytorch），部署环境均经过测试

### ONNX 部署

### OpenVINO 部署

```shell
python3 infer.py
```


### TensorRT 部署

TensorRT 版本间差异较大，需要根据版本进行调整，根据 CUDA 版本下载 [TensorRT](https://developer.nvidia.com/tensorrt) 和 [cuDNN](https://developer.nvidia.com/cudnn)

#### TensorRT 10.0 部署

> 「当前」版本组合为 2024.5.31 最新版本

测试环境如下：

- NVIDIA GeForce RTX 4090(24217MiB)*2
- **Ubuntu 22.04** (python3.10)
- **CUDA 12.1**: 虽然「当前」最新 CUDA 为 12.5，但是「当前」的 Pytorch(stable) 仅支持 12.1
- **TensorRT 10.0.1**: [*TensorRT 10.0 GA for Linux x86_64 and CUDA 12.0 to 12.4 TAR Package*](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz)
<!-- - **cuDNN 9.1.1**: [*cuDNN 9.1.1 (May 2024)*](https://docs.nvidia.com/deeplearning/cudnn/latest/release-notes.html#cudnn-9-1-1) -->



- **最简化安装如下**

如果顺利，使用 pip 安装完成后可以直接运行，大部分情况下可以采用如下办法
```shell
pip install pyyaml opencv-python \
    tensorrt==10.0.1 \
    cuda-python==12.1.0
# pip install nvidia-cudnn-cu12==9.1.1 # 经测试，非必要安装 # [!code --]
```


<!--  
- **TAR 包安装如下**

某些情况下，pip安装会导致安装不完全，需要从官方下载安装包

解压缩 TensorRT TAR 包
```shell
tar -xvf TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
```

向 `~/.bashrc` 添加环境变量（确保 CUDA 环境已经配置好），注意 `xxx_HOME` 的实际路径:
```shell
export TENSORRT_HOME=$HOME/program/TensorRT-10.0.1.6
export PATH=$TENSORRT_HOME/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH
```

安装 TensorRT Python 包
```shell
# python version 3.10
pip install $TENSORRT_HOME/python/tensorrt-10.0.1-cp310-none-linux_x86_64.whl --force-reinstall
```

该版本 TensorRT 的推理依赖于 `cuda-python`
```shell
pip install cuda-python==12.1.0
```

该版本 TensorRT 的推理依赖于 `nvidia-cudnn-cu12` ([*Installing cuDNN with Pip*](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#installing-cudnn-with-pip))，如果已经安装了 pytorch-gpu，可以跳过这一步，因为 pytorch-gpu 已经包含了 cuDNN
```shell
pip install nvidia-cudnn-cu12=9.1.1.17
``` -->
 

#### TensorRT 8.6 部署

> 当前版本组合为 2023.7.30 最新版本

测试环境如下：

- NVIDIA GeForce RTX 4090(24217MiB)*2
- **Ubuntu 22.04** (python3.10)
- **CUDA 12.1**
- **TensorRT 8.6**: [*TensorRT 8.6 GA for Linux x86_64 and CUDA 12.0 and 12.1 TAR Package*](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz)
- **cuDNN 8.9.7**: [*cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x*](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz/) (该版本貌似也可以 pip 安装，自行测试)

- **最简化安装如下**

如果顺利，使用 pip 安装完成后可以直接运行，大部分情况下可以采用如下办法
```shell
pip install pyyaml opencv-python \
    tensorrt==8.6.1 \
    cuda-python==12.1.0 \
    nvidia-cudnn-cu12==8.9.2.26
```
 
- **TAR 包安装如下**

某些情况下，pip安装会导致安装不完全，需要从官方下载安装包

解压缩 TensorRT 和 cuDNN TAR 包
```shell
tar -xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
```

向 `~/.bashrc` 添加环境变量（确保 CUDA 环境已经配置好），注意 `xxx_HOME` 的实际路径 :
```shell
export CUDNN_HOME=$HOME/program/cudnn-linux-x86_64-8.9.7.29_cuda12-archive
export TENSORRT_HOME=$HOME/program/TensorRT-8.6.1.6
export PATH=$TENSORRT_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$TENSORRT_HOME/lib:$LD_LIBRARY_PATH
```

安装 TensorRT Python 包
```shell
# python version 3.10
pip install $TENSORRT_HOME/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl --force-reinstall
```

该版本 TensorRT 的推理依赖于 `cuda-python`
```shell
pip install cuda-python==12.1.0
```

`nvidia-cudnn-cu12==8.9.2.26` ([*Installing cuDNN with Pip*](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#installing-cudnn-with-pip)) 可以 pip 安装，也可以找到
```shell
pip install nvidia-cudnn-cu12==8.9.2.26
```
 

 
#### TensorRT 推理

在 Yolov5 项目目录下执行
```shell
python3 export.py \
    --weights $MODEL_PATH \
    --data data/coco128.yaml \
    --simplify --device 0,1 --include engine   
```
- `--device` 指定 GPU 设备，有多少张卡就写多少张卡，用逗号分隔

修改 `infer.py` 文件，指定模型路径
```python
## -- TensorRT
args.model=".cache/yolov5/yolov5s.engine"
detector = backends.TensorRTBackend()
```


推理速度

| Tensor 版本 | 模型    | 推理时间(ms) | 设备           |
| ----------- | ------- | ------------ | -------------- |
| 10.0(.1.6)  | yolov5s | 3.26 ~ 3.72  | RTX 4090 (24G) |
| 8.6(.1.6)   | yolov5s | 3.40 ~ 3.85  | RTX 4090 (24G) |