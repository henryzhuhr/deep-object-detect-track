---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# 模型训练


## 下载预训练模型

<!--@include: ./download-pretrian.md-->

## 训练模型

提供一个训练脚本 `scripts/train.sh`，复制一份到项目目录下进行自定义修改（推荐）

```shell
cp scripts/train.sh scripts/train.custom.sh
```



修改训练脚本 `train.custom.sh` 中的参数，在 `Set Training Variables` 中设置
- `TRAIN_DATA_DIR`: (✅重要) 训练数据路径，例如 `$HOME/data/bottle-organized`
- `TRAIN_DEVICE`: (✅重要) 这里默认设置为 `cpu`；如果有 GPU 可以设置为 `0` 或者 `1`；多个 GPU 用逗号分隔，如果单 GPU 能够满足需求，建议使用单 GPU
- `MODEL_NAME`: 训练模型，用以加载 `--cfg` 参数，默认为 `yolov5s`
- `PRETRAINED_MODEL`: 上面下载好的预训练模型
- `EPOCHS`: epoch
- `BATCH_SIZE`: batch size


修改完成后，运行脚本开始训练
```shell
bash scripts/train.custom.sh
#zsh scripts/train.custom.sh # zsh
```

