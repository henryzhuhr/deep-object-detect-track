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

function print_help() {
    echo "Usage: $0 [--model=model1,model2,...]"
    echo ""
    echo "Options:"
    echo "  --model=<model_name>[,<model_name>,...]   specify one or more models to download (e.g., yolov5s,yolov5m)"
    echo "                                           if not specified, all models will be downloaded"
    echo ""
    echo "Supported models:"
    printf '  %s\n' "${all_model_list[@]}"
    exit 1
}

function query_model_list() {
    local model_list=()
    for model in "${all_model_list[@]}"; do
        model_list+=("$model,")
    done
    echo "supported model: ${model_list[*]}"
}

# 解析命令行参数
model_names=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model=*)
            IFS=',' read -r -a model_names <<< "${1#*=}"
            ;;        --query-model-list)
            query_model_list
            exit 0
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help to see available options."
            exit 1
            ;;
    esac
    shift
done

if [ ${#model_names[@]} -eq 0 ]; then
    log_info "No model specified, downloading all models..."
    model_list=("${all_model_list[@]}")
else
    model_list=()
    for m in "${model_names[@]}"; do
        if printf '%s\n' "${all_model_list[@]}" | grep -q "^${m}$"; then
            model_list+=("$m")
        else
            echo "Supported models:"
            printf '  %s\n' "${all_model_list[@]}"
            log_fatal "Model '$m' is not supported."
        fi
    done
fi

if [[ -z "$CACHE_DIR" ]]; then
    CACHE_DIR="${HOME}/.cache"
fi

weights_dir="$CACHE_DIR/yolov5/models"
[ ! -d "${weights_dir}" ] && mkdir -p "${weights_dir}"

log_info "Downloading YOLOv5 (${model_list[*]}) weights..."
for model_name in "${model_list[@]}"; do
    url=https://github.com/ultralytics/yolov5/releases/download/${tag_name}/${model_name}.pt
    if [[ -f "${weights_dir}/${model_name}.pt" ]]; then
        log_info "${model_name}.pt already exists in ${weights_dir}/${model_name}.pt, skipping download."
        continue
    fi
    log_info "Downloading ${url} ..."
    wget --no-clobber --continue -c "${url}" -P "${weights_dir}" #--no-verbose
    if [[ -f "${weights_dir}/${model_name}.pt" ]]; then
        log_success "Downloaded ${model_name} successfully. saved to ${weights_dir}/${model_name}.pt"
    else
        log_fatal "Failed to download ${model_name}."
    fi
done

ls -lh "${weights_dir}"
