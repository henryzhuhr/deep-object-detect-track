---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# YOlOv5 项目

## 获取源码

获取 [ultralytics](https://github.com/ultralytics/ultralytics) 项目源码

```bash
cd ~/project
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
```

建议固定某一个版本，查看当前存在的tag版本，并选择一个合适的版本进行切换：

```bash
git tag
ttag=vX.Y.Z # 替换为你选择的版本(Z建议设置为0)，例如 ttag=v8.3.0
git switch -c tag-${ttag} tags/${ttag}
```

## 获取模型

模型版本一般为 `vX.Y.0`

这里获取 YOLO11 系列模型

```bash:line-numbers
ttag=vX.Y.Z # ttag=v8.3.0
models=(yolo11n yolo11s yolo11m yolo11l yolo11x) # 根据需要修改需要下载的模型
source .env && mkdir -p $YOLO_MODEL_DIR && for model in "${models[@]}"; do { [ ! -f "$local_file" ] && curl -L -o ${YOLO_MODEL_DIR}/${model}.pt https://github.com/ultralytics/assets/releases/download/${ttag}/${model}.pt; } done
ls -alh ${YOLO_MODEL_DIR}
```

## 配置 Python 环境

```bash
uv sync
```

如果报错，尝试修改 `pyproject.toml` 文件的 `requires-python = ">=3.8"` 为 `requires-python = "==3.10"`，然后重新执行 `uv sync`。

此外，如果涉及模型导出，可能需要手动安装一些依赖：

```bash
uv sync --extra export --extra dev
```

> macOS 可能需要注释掉 `tensorflow` 相关的依赖项

或者直接安装全部的依赖

```bash
uv sync --all-extras
```

## 导出模型

在 `tests/__init__.py` 文件中配置你要导出的模型

### 导出 ONNX 模型

```bash
pytest tests/test_exports.py::test_export_onnx -v
```
