
PROJECT_HOME=$(pwd)
yolov5_path=$PROJECT_HOME/projects/yolov5

YOLO_PRETRAINED_MODE_DIRL=$PROJECT_HOME/.cache/yolov5
EXPORT_MODEL=yolov5s.pt

MODEL_PATH=$YOLO_PRETRAINED_MODE_DIRL/$EXPORT_MODEL
# MODEL_PATH=.. # 取消注释，自定义模型路径

cd $yolov5_path

# export the model to the onnx format

python3 export.py \
    --weights $MODEL_PATH \
    --data data/coco128.yaml \
    --simplify --include onnx 

python3 export.py \
    --weights $MODEL_PATH \
    --data data/coco128.yaml \
    --simplify --include openvino 

python3 export.py \
    --weights $MODEL_PATH \
    --data data/coco128.yaml \
    --simplify --device 0,1 --include engine

# 0,1 单 GPU 貌似报错，电脑有多少就得给多少
# CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)
# CUDA:1 (NVIDIA GeForce RTX 4090, 24217MiB)