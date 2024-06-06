
if [ ! -f "scripts/base.sh" ]; then
    echo "scripts/base.sh not found"
    exit 1
fi
source scripts/base.sh


# =============== Set Training Variables ================

# -- Yolov5 project path / 项目路径
yolov5_path=$PROJECT_HOME/projects/yolov5

DATASET_CONFIG=$yolov5_path/data/coco128.yaml

# -- Path to exported model / 需要导出模型的路径
EXPORTED_MODEL_PATH=$PROJECT_HOME/.cache/yolov5/yolov5s.pt

# -- 导出 TensorRT 的参数
# 如果是单卡 设置为 0
# CUDA:0 (NVIDIA GeForce RTX 3080, 12117MiB)
# 如果是双卡 设置为 0,1
# CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)
# CUDA:1 (NVIDIA GeForce RTX 4090, 24217MiB)
# 电脑有多少显卡就得给多少
TRT_EXPORTED_DEVICE="0,1" # Multiple GPUs
# TRT_EXPORTED_DEVICE="0"   # Single GPU

# =======================================================

cd $yolov5_path

python3 export.py \
    --weights $EXPORTED_MODEL_PATH \
    --data $DATASET_CONFIG \
    --simplify --include onnx 


python3 export.py \
    --weights $EXPORTED_MODEL_PATH \
    --data $DATASET_CONFIG \
    --simplify --include openvino 


python3 export.py \
    --weights $EXPORTED_MODEL_PATH \
    --data $DATASET_CONFIG \
    --simplify --include engine --device $TRT_EXPORTED_DEVICE
