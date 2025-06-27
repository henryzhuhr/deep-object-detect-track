#!/bin/bash

if [ ! -f "scripts/base.bash" ]; then
    echo "scripts/base.bash not found"
    exit 1
fi
# shellcheck source=scripts/base.bash
source scripts/base.bash

tag_name=v7.0
all_model_list=(
    yolov5n
    yolov5n6
    yolov5s
    yolov5s6
    yolov5m
    yolov5m6
    yolov5l
    yolov5l6
    yolov5x
    yolov5x6
)

weights_dir=$CACHE_DIR/yolov5
[ ! -d "${weights_dir}" ] && mkdir -p "${weights_dir}"

model_list=("$1")

if [ ${#model_list[@]} -eq 0 ]; then
    model_list=("${all_model_list[@]}")
fi

log_info "Downloading YOLOv5 (${model_list[*]}) weights..."
echo ""

for model_name in "${model_list[@]}"; do
    url=https://github.com/ultralytics/yolov5/releases/download/${tag_name}/${model_name}.pt
    log_info "Downloading ${url} ..."
    wget -c "${url}" -P "${weights_dir}" # --no-verbose
    log_success "Downloaded successfully: ${weights_dir}/${model_name}.pt"
    echo
done
