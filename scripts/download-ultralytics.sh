#!/bin/bash
ttag=vX.Y.Z # 
ttag=v8.3.0
models=(yolo11n yolo11s yolo11m yolo11l yolo11x) # 根据需要修改需要下载的模型

if [ "$ttag" = "vX.Y.Z" ]; then
  echo "Please set the ttag variable to the desired version, e.g., v8.3.0" && exit 1
fi

source .env

if [ -z "$YOLO_MODEL_DIR" ]; then
  YOLO_MODEL_DIR=~/.cache/ultralytics
fi
mkdir -p "$YOLO_MODEL_DIR"

for model in "${models[@]}"; do
  local_file="${YOLO_MODEL_DIR}/${model}.pt"
  if [ ! -f "$local_file" ]; then
    curl -L -o "${YOLO_MODEL_DIR}/${model}.pt" "https://github.com/ultralytics/assets/releases/download/${ttag}/${model}.pt" && \
    echo "Downloaded $model to $local_file"
  fi
done

echo "downloaded models in ${YOLO_MODEL_DIR}:"
ls -alh "${YOLO_MODEL_DIR}"