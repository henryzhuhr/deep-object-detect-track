if [ ! -f "scripts/base.sh" ]; then
    echo "scripts/base.sh not found"
    exit 1
fi
source scripts/base.sh

tag_name=v1.1.0
base_url=https://github.com/HenryZhuHR/deep-object-detect-track/releases/download/${tag_name}


weights_dir=$CACHE_DIR/yolov5
[ ! -d ${weights_dir} ] && mkdir -p ${weights_dir}

wget -c ${base_url}/coco.yaml -P ${weights_dir}
wget -c ${base_url}/yolov5s.onnx -P ${weights_dir}


ov_dir=${weights_dir}/yolov5s_openvino_model
[ ! -d ${ov_dir} ] && mkdir -p ${ov_dir}

wget -c ${base_url}/yolov5s.bin -P ${ov_dir}
wget -c ${base_url}/yolov5s.xml -P ${ov_dir}

drink_dir=$CACHE_DIR/yolov5_drink
[ ! -d ${drink_dir} ] && mkdir -p ${drink_dir}
wget -c ${base_url}/drink.yaml -P ${drink_dir}
wget -c ${base_url}/yolov5s-drink.onnx -P ${drink_dir}
wget -c ${base_url}/yolov5s-drink.xml -P ${drink_dir}
wget -c ${base_url}/yolov5s-drink.bin -P ${drink_dir}
