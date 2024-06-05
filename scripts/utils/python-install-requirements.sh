
print_info "Using Python: $(which python3)"
print_info "Installing Python requirements..."

if [ -z "${verbose}" ]; then
    verbose=false
fi

if [ "$verbose" = false ]; then
    PIPQ="-q"
else
    PIPQ=""
fi

# ================== pip ==================
print_info "Upgrading pip..."
python3 -m pip install $PIPQ --upgrade pip

# ================== PyTorch ==================
# install pytorch according to the CUDA version
if [ ! -z "${CUDA_VERSION}" ]; then
    print_info "CUDA_VERSION is set to $CUDA_VERSION. Installing PyTorch with CUDA support."
    python3 -m pip install torch torchvision $PIPQ --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d '.')
else
    print_warning "CUDA_VERSION is not set. Installing CPU version of PyTorch."
    python3 -m pip install torch torchvision $PIPQ --index-url https://download.pytorch.org/whl/cpu
fi

# ================== requirements.txt ==================
print_info "Installing other requirements..."
mkdir -p $CACHE_DIR/yolov5
if [ ! -f "$CACHE_DIR/yolov5/requirements.txt" ]; then
    cp projects/yolov5/requirements.txt $CACHE_DIR/yolov5/requirements.txt
fi
python3 -m pip install $PIPQ -r $CACHE_DIR/yolov5/requirements.txt
python3 -m pip install $PIPQ -r requirements/requirements.train.txt


# ================== ONNX Runtime ==================
# if [ ! -z "${CUDA_VERSION}" ]; then
#     python3 -m pip install $PIPQ onnxruntime-gpu
# else
#     python3 -m pip install $PIPQ onnxruntime
# fi
python3 -m pip install $PIPQ onnxruntime


# ================== OpenVINO ==================
# if [ -d "$INTEL_OPENVINO_DIR" ]; then
#     print_info "OpenVINO found in $INTEL_OPENVINO_DIR"
#     print_info "Installing OpenVINO Python requirements from $INTEL_OPENVINO_DIR/python/requirements.txt"
#     python3 -m pip install $PIPQ -r $INTEL_OPENVINO_DIR/python/requirements.txt
# else
#     print_warning "OpenVINO not found, skipping OpenVINO Python requirements installation"
#     print_info "Installing OpenVINO Python requirements from pip"
#     python3 -m pip install $PIPQ openvino-dev
# fi
python3 -m pip install $PIPQ openvino-dev

# freeze the requirements 
# python3 -m pip list --format=freeze > requirements.version.txt
