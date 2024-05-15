
print_info "Using Python: $(which python3)"
print_info "Installing Python requirements..."

python3 -m pip install --upgrade pip

# install pytorch according to the CUDA version
if [ ! -z "${CUDA_VERSION}" ]; then
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d '.')
else
    python3 -m pip install torch torchvision
    print_warning "CUDA_VERSION is not set. Installing CPU version of PyTorch."
fi


mkdir -p .cache/yolov5
if [ ! -f ".cache/yolov5/requirements.txt" ]; then
    cp projects/yolov5/requirements.txt .cache/yolov5/requirements.txt
fi
python3 -m pip install -r .cache/yolov5/requirements.txt
python3 -m pip install -r requirements.txt

python3 -m pip install onnx
python3 -m pip install onnx-simplifier

if [ ! -z "${CUDA_VERSION}" ]; then
    python3 -m pip install onnxruntime-gpu
else
    python3 -m pip install onnxruntime
fi

if [ -d "$INTEL_OPENVINO_DIR" ]; then
    print_info "OpenVINO found in $INTEL_OPENVINO_DIR"
    print_info "Installing OpenVINO Python requirements from $INTEL_OPENVINO_DIR/python/requirements.txt"
    python3 -m pip install -r  $INTEL_OPENVINO_DIR/python/requirements.txt
else
    print_warning "OpenVINO not found"
    print_info "Installing OpenVINO Python requirements from pip"
    python3 -m pip install openvino-dev
fi


# freeze the requirements 
# python3 -m pip list --format=freeze > requirements.version.txt
