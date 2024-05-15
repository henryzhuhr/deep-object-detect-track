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
conda create -p .env/deep-object-detect-track python=3.10 -y
conda activate ./.env/deep-object-detect-track
# 全局安装环境
conda create -n deep-object-detect-track python=3.10 -y
conda activate deep-object-detect-track
```
> Python 版本选择 3.10 是因为 Ubuntu 22.04 默认安装的 Python 版本是 3.10

### 安装依赖

1. 安装 PyTorch

参考官网 [INSTALL PYTORCH](https://pytorch.org/get-started/locally/) 选择配置安装 PyTorch

```shell
python3 -m pip3 install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121
```
> 链接最后的 `cu121` 是需要根据系统的 CUDA 版本进行选择

接下来安装其他依赖

```shell
python3 -m pip install -r projects/yolov5/requirements.txt
python3 -m pip install -r requirements.txt
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

### ONNX 部署
### OpenVINO 部署

```shell
python3 infer.py
```

