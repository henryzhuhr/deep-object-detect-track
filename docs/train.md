---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# 模型训练


## 下载预训练模型

<!--@include: ./download-pretrian.md-->


## 训练模型（手动设置参数）


```shell
python train.py \
    --data ~/data/drink-organized/dataset.yaml \
    --cfg models/yolov5s.yaml --weights ../../.cache/yolov5/yolov5s.pt \
    --epochs 10 --batch 2 \
    --device 0 --workers 8 --cache disk \
    --project ../../tmp/train
```

- `--data`: 数据集配置文件，yaml 格式
- `--cfg`: 模型配置文件，yaml 格式，需要和预训练模型参数 `--weights` 对应
- `--device`: 如果有 GPU 可以设置为 `0` 或者 `1`；多个 GPU 用逗号分隔，如果单 GPU 能够满足需求，建议使用单 GPU
- `--batch`: batch size，最重要的参数，建议从 1/2 开始尝试，以避免内存不足，而不要一开始就设置一个很大的值导致系统崩溃
- `--cache`: 缓存类型，`ram` 或者 `disk`，可以加快训练速度，如果内存足够，建议使用 `ram` ，也可以不使用该参数而不缓存数据

如果已经有了训练好的模型，可以继续训练，使用 `--resume` 参数从上次训练的 `last.pt` 模型继续训练

```shell

## 训练模型（使用提供的脚本）

提供一个训练脚本 `scripts/train.sh`，复制一份到项目目录下进行自定义修改（推荐）

```shell
cp scripts/train.sh scripts/train.custom.sh
```
修改训练脚本 `train.custom.sh` 中的参数，在 `Set Training Variables` 中设置
- `TRAIN_DATA_DIR`: (✅重要) 训练数据路径，例如 `$HOME/data/bottle-organized`
- `TRAIN_DEVICE`: (✅重要) 如果有 GPU 可以设置为 `0` 或者 `1`；多个 GPU 用逗号分隔，如果单 GPU 能够满足需求，建议使用单 GPU
- `MODEL_NAME`: 训练模型，用以加载 `--cfg` 参数，默认为 `yolov5s`
- `PRETRAINED_MODEL`: 上面下载好的预训练模型
- `EPOCHS`: epoch
- `BATCH_SIZE`: batch size


修改完成后，运行脚本开始训练
```shell
bash scripts/train.custom.sh
#zsh scripts/train.custom.sh # zsh
```

