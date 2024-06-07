---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# 模型部署

## 准备模型

如果训练了自定义模型，或者下载了预训练模型，可以跳过这一步

<!--@include: ./download-pretrian.md-->

同时，也提供了一些转化好的模型（从 ultralytics/yolov5(v7.0) 的 [yolov5](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s) 导出），可以直接使用进行部署

```shell
bash scripts/download-release.sh
#zsh scripts/download-release.sh # zsh
```

## 导出模型

提供一个导出脚本 `scripts/train.sh`，复制一份到项目目录下进行自定义修改（推荐）

```shell
cp scripts/export-yolov5.sh scripts/export-yolov5.custom.sh
```

查看脚本 `scripts/export-yolov5.custom.sh` ，根据项目需求修改参数后执行

```shell
bash scripts/export-yolov5.custom.sh
#zsh scripts/export-yolov5.custom.sh # zsh
```


## 部署模型

部署模型请单独创建虚拟环境，并根据部署中提到的依赖进行**最小化安装**，而避免引入过多不必要的包 （例如 pytorch），部署环境均经过测试

### ONNX 部署

修改 `scripts/variables.custom.sh` 文件中 `ENV_NAME` 如下

```shell
# export ENV_NAME="" # -- Uncomment to customize the environment name # [!code --]
export ENV_NAME="deploy-onnx" # [!code ++]
export ENV_PATH=$BASE_ENV_PATH/.env/$ENV_NAME
```

然后执行脚本创建虚拟环境，并激活

::: code-group

```shell [使用 venv]
bash scripts/create-python-env.sh -e venv
#zsh scripts/create-python-env.sh -e venv # zsh
```

```shell [使用 conda]
bash scripts/create-python-env.sh -e conda
#zsh scripts/create-python-env.sh -e conda # zsh
```

:::

根据上述脚本运行输出激活环境

```shell
- [INFO] Run command below to activate the environment:
... # 复制这里出现的激活命令并执行
```

手动安装依赖

```shell
pip install -r requirements/requirements.txt
pip install onnxruntime # CPU 版本
# pip install onnxruntime-gpu # GPU 版本
```

修改 `infer.py` 文件，指定模型路径
```python
## ------ ONNX ------
onnx_backend = backends.ONNXBackend
print("-- Available devices:", providers := onnx_backend.SUPPORTED_DEVICES)
detector = onnx_backend(
    device=providers, inputs=["images"], outputs=["output0"]
)
```

然后执行推理脚本

```shell
python infer.py --model .cache/yolov5/yolov5s.onnx
```


### OpenVINO 部署

<!-- 在运行 Openvino 前，请安装好 OpenVINO ，并手动激活环境   
```shell
source <openvino_install_path>/setupvars.sh
source /opt/intel/openvino_2024/setupvars.sh
``` -->

修改 `scripts/variables.custom.sh` 文件中 `ENV_NAME` 如下

```shell
# export ENV_NAME="" # -- Uncomment to customize the environment name # [!code --]
export ENV_NAME="deploy-ov" # [!code ++]
export ENV_PATH=$BASE_ENV_PATH/.env/$ENV_NAME
```

然后执行脚本创建虚拟环境，并激活

::: code-group

```shell [使用 venv]
bash scripts/create-python-env.sh -e venv
#zsh scripts/create-python-env.sh -e venv # zsh
``` 

```shell [使用 conda]
bash scripts/create-python-env.sh -e conda
#zsh scripts/create-python-env.sh -e conda # zsh
```

:::

根据上述脚本运行输出激活环境

```shell
- [INFO] Run command below to activate the environment:
... # 复制这里出现的激活命令并执行
```

手动安装依赖

```shell
pip install -r requirements/requirements.txt
pip install openvino-dev
```
<!-- # [ -d "$INTEL_OPENVINO_DIR" ] && pip install -r $INTEL_OPENVINO_DIR/python/requirements.txt || echo "Please install OpenVINO Toolkit" -->

修改 `infer.py` 文件，指定模型路径
```python
## ------ ONNX ------
ov_backend = backends.OpenVINOBackend
print("-- Available devices:", ov_backend.query_device())
detector = ov_backend(device="AUTO") # [!code --]
detector = ov_backend(device="GPU")  # 指定使用 GPU 推理 # [!code ++]
```

然后执行推理脚本

```shell
python infer.py --model .cache/yolov5/yolov5s_openvino_model/yolov5s.xml
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

修改 `infer.py` 文件
```python
## -- TensorRT
detector = backends.TensorRTBackend()
```

执行推理
```shell
python infer.py --model .cache/yolov5/yolov5s.engine
```


推理速度

| Tensor 版本 | 模型    | 推理时间(ms) | 设备           |
| ----------- | ------- | ------------ | -------------- |
| 10.0(.1.6)  | yolov5s | 3.26 ~ 3.72  | RTX 4090 (24G) |
| 10.0(.1.6)  | yolov5s | 10.41 ~ 10.7  | RTX 3050 (8G) |
| 8.6(.1.6)   | yolov5s | 3.40 ~ 3.85  | RTX 4090 (24G) |