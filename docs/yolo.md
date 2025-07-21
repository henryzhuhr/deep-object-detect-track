---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# YOlOv5 项目

## 获取源码

获取 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 源码

```bash
cd ~/project
git clone https://github.com/ultralytics/yolov5
cd yolov5
```

## 配置 Python 环境

项目需要修改 `pyproject.toml` 文件

的 `export` 部分，只保留 `onnx` 相关的依赖，注释其他依赖项
注释 `export` 部分所有的依赖项，按照实际需求（系统和硬件）单独添加

```toml
export = [
    "onnx>=1.12.0", # ONNX export
]
```

如果是 TensorRT 导出，还需要添加 `tensorrt` 相关的依赖：

```toml
export = [
    "onnx>=1.12.0", # ONNX export
    "nvidia-pyindex",
    "nvidia-tensorrt",
] 

添加以下内容：

```toml
[tool.setuptools.packages.find]
include = ["data", "models", "segment", "classify"]
```

然后执行：

```bash
export UV_PYTHON_PREFERENCE=only-system 
export UV_HTTP_TIMEOUT=3600
# uv sync
uv sync --no-install-package tensorrt --no-install-package pyyaml
WHEEL=$(ls dist/inference_exp-*.whl) && uv pip install --system --no-deps "${WHEEL}"
```

如果需要训练模型或者调用CUDA设备，需要执行命令检查：

```bash
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd,) && echo "import torch; print(f'PyTorch find {torch.cuda.device_count()} GPUs:');
for i in range(torch.cuda.device_count()):
    print(f' ✅ Device {i}: {torch.cuda.get_device_properties(i)}')" | uv run -
```

## 下载预训练模型

这一步需要在当前项目中进行，当前项目下提供了获取 yolov5 项目资源的脚本，下载模型：

```bash
# 下载所有模型
bash scripts/download-yolov5.bash
# 仅下载指定模型
bash scripts/download-yolov5.bash --model=yolov5s,yolov5m
```

上述资源文件下载的路径 `${HOME}/.cache/yolov5` 可以通过修改 `scripts/variables.custom.sh` 文件中的 `CACHE_DIR` 变量来修改
