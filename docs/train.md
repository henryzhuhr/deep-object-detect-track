---
lastUpdated: true
editLink: true
footer: true
outline: deep
---

# 模型训练


## 下载预训练模型

下载预训练模型

```bash
bash scripts/download-yolov5.sh
bash scripts/download-yolov5.sh yolov5s # 仅下载单个模型
```

运行脚本开始训练

```bash
bash scripts/train.sh
```

在 `Set Training Variables` 项目中设置各个变量
- `TRAIN_DATA_DIR`: 训练数据路径，例如 `$HOME/data/bottle-organized`
- `MODEL_NAME`: 训练模型，用以加载 `--cfg` 参数
- `PRETRAINED_MODEL`: 上面下载好的预训练模型
- `EPOCHS`: epoch
- `BATCH_SIZE`: batch size