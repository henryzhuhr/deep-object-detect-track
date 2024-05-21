tag_name=v7.0
all_model_list=(
    yolov5n
    yolov5s
    yolov5m
    yolov5l
    yolov5x
    yolov5n
)

weights_dir=./.cache/yolov5
mkdir -p ${weights_dir}

model_list=($1)

if [ ${#model_list[@]} -eq 0 ]; then
    model_list=(${all_model_list[@]})
fi

for model_name in ${model_list[@]}; do
    url=https://github.com/ultralytics/yolov5/releases/download/${tag_name}/${model_name}.pt
    echo "Downloading ${model_name}.pt from ${url}"
    wget -c ${url} -P ${weights_dir}
done
