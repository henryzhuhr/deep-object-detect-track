#!/bin/bash

source ".env"

model="yolo11s"

if [ -z "$YOLO_MODEL_DIR" ]; then
    echo "Error: YOLO_MODEL_DIR is not set in .env file."
    exit 1
fi


trtexec \
    --onnx="${YOLO_MODEL_DIR}/${model}.onnx" \
    --saveEngine="${YOLO_MODEL_DIR}/${model}.engine" \
    --best \
    --stronglyTyped \
    --verbose