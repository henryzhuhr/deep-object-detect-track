#!/bin/bash
set -e

model_list=(
    "yolo11n"
    "yolo11s"
    "yolo11m"
)

source ".env"

if [ -z "$YOLO_MODEL_DIR" ]; then
    echo "Error: YOLO_MODEL_DIR is not set in .env file."
    exit 1
fi


for model in "${model_list[@]}"; do
    if [ ! -f "${YOLO_MODEL_DIR}/${model}.onnx" ]; then
        echo "Error: ONNX model ${model}.onnx not found in ${YOLO_MODEL_DIR}."
        exit 1
    fi

    echo "Converting ${model}.onnx to TensorRT engine..."
    trtexec \
        --onnx="${YOLO_MODEL_DIR}/${model}.onnx" \
        --saveEngine="${YOLO_MODEL_DIR}/${model}.engine" \
        --best \
        --stronglyTyped \
        --verbose
done

